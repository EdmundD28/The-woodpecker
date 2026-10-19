import copy
import unittest
import numpy as np

from cnn_analysis.core import config
from cnn_analysis.quality import (EXCLUDED, FEATURE_NAMES, position_group, partition,
                                  features, fit, calibrate, novelty_score, predict)
from cnn_analysis.protocol import build_sample_id


class QualityTests(unittest.TestCase):
    def record(self,point=1,strike=1,stick='WOOD01'):
        t=np.maximum((np.arange(2400)-256)/8000,0)
        signal=650*np.exp(-22*t)*np.sin(2*np.pi*(450+strike)*t)
        return dict(protocol_version='WOOD_IMPACT_V1',sample_id=build_sample_id(stick,1,0,point,strike),
                    stick_id=stick,experiment_batch=1,reclamp_batch=0,impact_point_id=point,
                    strike_id=strike,sample_rate_hz=8000,sample_count=2400,anomaly_flags=[],
                    raw=np.rint(2000+signal).astype(int).tolist())

    def pool(self):
        return [self.record(p,h,stick) for stick in ['WOOD01','WOOD02'] for p in [1,2,3] for h in range(1,51)]

    def test_named_abnormals_excluded_from_all_normal_partitions(self):
        pool=self.pool(); parts=partition(pool)
        self.assertEqual({k:len(v) for k,v in parts.items()},
                         {'train':176,'calibration':58,'normal_test':61,'abnormal_test':5})
        ids={k:{r['sample_id'] for r in rows} for k,rows in parts.items()}
        self.assertEqual(ids['abnormal_test'],EXCLUDED)
        for name,values in ids.items():
            for other,second in ids.items():
                if name!=other: self.assertFalse(values & second)
        self.assertEqual(ids,{k:{r['sample_id'] for r in rows} for k,rows in partition(pool).items()})
        for name in ['train','calibration','normal_test']:
            self.assertEqual({(r['stick_id'],r['impact_point_id']) for r in parts[name]},
                             {(s,p) for s in ['WOOD01','WOOD02'] for p in [1,2,3]})

    def test_features_do_not_depend_on_wood_identity_or_endpoint_number(self):
        r=self.record(); renamed=copy.deepcopy(r)
        renamed['stick_id']='UNKNOWN'; renamed['sample_id']=build_sample_id('UNKNOWN',1,0,3,1)
        renamed['impact_point_id']=3
        np.testing.assert_array_equal(features(r,config()),features(renamed,config()))
        self.assertEqual(len(features(r,config())),len(FEATURE_NAMES))
        self.assertEqual(position_group(1),position_group(3))
        self.assertNotEqual(position_group(1),position_group(2))

    def test_normal_calibration_uses_exact_held_out_distance_rank(self):
        parts=partition(self.pool()); model=fit(parts['train'],config())
        reference=copy.deepcopy(model['groups'])
        scores={name:sorted(novelty_score(r,model) for r in parts['calibration']
                            if position_group(r['impact_point_id'])==name) for name in reference}
        calibrate(model,parts['calibration'])
        self.assertEqual(model['groups']['endpoint']['threshold'],scores['endpoint'][37])
        self.assertEqual(model['groups']['middle']['threshold'],scores['middle'][18])
        for name,group in model['groups'].items():
            self.assertEqual(group['reference'],reference[name]['reference'])
            self.assertEqual(group['median'],reference[name]['median'])
            self.assertEqual(group['scale'],reference[name]['scale'])

    def test_distance_and_invalid_capture_check(self):
        parts=partition(self.pool()); model=calibrate(fit(parts['train'],config()),parts['calibration'])
        r=self.record(); group=model['groups']['endpoint']
        z=(features(r,config())-np.array(group['median']))/np.array(group['scale'])
        expected=sorted(float(np.sqrt(np.mean((np.array(row)-z)**2))) for row in group['reference'])[:5]
        self.assertAlmostEqual(novelty_score(r,model),sum(expected)/5)
        renamed=copy.deepcopy(r); renamed['stick_id']='OTHER'
        renamed['sample_id']=build_sample_id('OTHER',1,0,1,1)
        self.assertEqual(predict(r,model)['score'],predict(renamed,model)['score'])
        frozen=copy.deepcopy(model)
        for flag in ['CLIPPED','SAMPLE_LOSS','WEAK_SIGNAL']:
            bad=copy.deepcopy(r); bad['anomaly_flags']=[flag]
            self.assertTrue(predict(bad,model)['retest'])
        for value in [0,4095]:
            bad=copy.deepcopy(r); bad['raw'][100]=value
            self.assertEqual(predict(bad,model)['reason'],'INVALID_CAPTURE')
        weak=copy.deepcopy(r); weak['raw']=[2000]*2400
        self.assertTrue(predict(weak,model)['retest'])
        self.assertEqual(model,frozen)  # Evaluation never updates learned reference or threshold.


if __name__=='__main__': unittest.main()
