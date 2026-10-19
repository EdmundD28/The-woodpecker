import copy
import unittest
from unittest.mock import patch

import numpy as np

from cnn_analysis.release_predict import ReleasedPredictor
from pc_collector.protocol import build_sample_id


def captures(stick='UNKNOWN',batch=9):
    rows=[];t=np.arange(2400)/8000
    for point in [1,2,3]:
        signal=300*np.sin(2*np.pi*(400+100*point)*t)*np.exp(-np.maximum(t-.032,0)*18)
        raw=np.rint(1950+signal).astype(int).tolist()
        rows.append({'protocol_version':'WOOD_IMPACT_V1','sample_id':build_sample_id(stick,batch,0,point,1),
            'stick_id':stick,'experiment_batch':batch,'reclamp_batch':0,'impact_point_id':point,'strike_id':1,
            'sample_rate_hz':8000,'sample_count':2400,'anomaly_flags':[],'raw':raw})
    return rows


class ReleasePredictTests(unittest.TestCase):
    def test_endpoint_waveforms_can_exchange(self):
        rows=captures();swapped=copy.deepcopy(rows);swapped[0]['raw'],swapped[2]['raw']=swapped[2]['raw'],swapped[0]['raw']
        for seed in [42,123,2026]:
            predictor=ReleasedPredictor(seed)
            np.testing.assert_array_equal(predictor._frequency_scores(rows),predictor._frequency_scores(swapped))

    def test_identity_is_not_a_model_feature(self):
        predictor=ReleasedPredictor();a=captures();b=captures('UNSEEN_WOOD',11)
        np.testing.assert_array_equal(predictor._frequency_scores(a),predictor._frequency_scores(b))
        pa=predictor.predict(a);pb=predictor.predict(b)
        self.assertEqual(pa['status'],pb['status'])
        self.assertEqual([p['retest'] for p in pa['quality']],[p['retest'] for p in pb['quality']])

    def test_quality_rejection_stops_classification(self):
        predictor=ReleasedPredictor();rows=captures();rows[1]['raw'][500]=4095
        with patch.object(predictor,'_frequency_scores',side_effect=AssertionError('classification must not run')):
            result=predictor.predict(rows)
        self.assertEqual(result['status'],'RETEST');self.assertIn(2,result['retest_points']);self.assertNotIn('class',result)

    def test_malformed_capture_is_retested(self):
        rows=captures();rows[0]['raw']=rows[0]['raw'][:-1]
        result=ReleasedPredictor().predict(rows)
        self.assertEqual(result['status'],'RETEST');self.assertIn(1,result['retest_points'])

    def test_duplicate_position_is_rejected(self):
        rows=captures()
        with self.assertRaises(ValueError):ReleasedPredictor().predict([rows[0],rows[0],rows[2]])

    def test_unreleased_seed_cannot_load(self):
        with self.assertRaises(ValueError):ReleasedPredictor(999)


if __name__=='__main__':unittest.main()
