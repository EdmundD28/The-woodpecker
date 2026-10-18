import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from luo_analysis.core import config, preprocess, dump
from luo_analysis.network import CNN
from luo_analysis.triplet import (FORMAT, POINTS, inputs, probabilities, combinations,
                                  assemble, partition, contract, predict, export_triplet, features)
from pc_collector.protocol import build_sample_id


class TripletTests(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.c.update(classes=['WOOD01', 'WOOD02'], class_display_names=['木头1', '木头2'],
                      stick_labels={'WOOD01': 'WOOD01', 'WOOD02': 'WOOD02'})
        self.groups = [{'id': f'{label}-{i}', 'label': label} for label in range(2) for i in range(10)]

    def records(self):
        records = []
        for p in POINTS:
            t = np.maximum((np.arange(2400)-256)/8000, 0)
            raw = np.rint(2000+600*np.exp(-30*t)*np.sin(2*np.pi*(400+p*100)*t)).astype(int).tolist()
            records.append(dict(protocol_version='WOOD_IMPACT_V1', stick_id='WOOD01',
                                sample_id=build_sample_id('WOOD01', 1, 0, p, 1),
                                experiment_batch=1, reclamp_batch=0, impact_point_id=p, strike_id=1,
                                sample_rate_hz=8000, sample_count=2400, anomaly_flags=[], raw=raw))
        return records

    def test_partition_and_combinations_preserve_positions_and_labels(self):
        parts = partition(self.groups, self.c)
        self.assertEqual({k:len(v) for k,v in parts.items()}, {'train':12,'validation':4,'test':4})
        self.assertFalse(set(parts['train']) & set(parts['test']))
        self.assertFalse(set(parts['validation']) & set(parts['test']))
        rows = combinations(self.groups, parts['test'])
        self.assertEqual(len(rows), 16)  # 2 specimens * 2**3 combinations.
        time = np.empty((20, 2400, 3), dtype=np.float32)
        for i in range(20):
            for p in range(3): time[i,:,p] = i*10+p
        x, y = assemble(time, self.groups, rows)
        for sample, row, label in zip(x, rows, y):
            for p in range(3):
                self.assertEqual(sample[0,p], row[p]*10+p)
                self.assertEqual(self.groups[row[p]]['label'], label)
        with self.assertRaises(ValueError): assemble(time, self.groups, np.array([[0,10,0]]))

    def test_three_channel_gradients_and_reversal_invariance(self):
        for mode in ['time', 'spectrum']:
            c = {**self.c,'input_mode':mode}
            net = CNN({**c,'channels':3})
            x = inputs(np.concatenate([preprocess(r,c) for r in self.records()],axis=1)[None,:,:],c)
            np.testing.assert_array_equal(probabilities(net,x), probabilities(net,x[:,:,::-1]))
            loss, gradients = net.gradients(x,np.array([0]))
            self.assertTrue(np.isfinite(loss))
            idx = np.unravel_index(np.abs(gradients['w0']).argmax(), gradients['w0'].shape)
            original = net.p['w0'][idx].copy(); eps=.001
            net.p['w0'][idx]=original+eps; plus=net.gradients(x,np.array([0]))[0]
            net.p['w0'][idx]=original-eps; minus=net.gradients(x,np.array([0]))[0]
            net.p['w0'][idx]=original
            self.assertAlmostEqual(float(gradients['w0'][idx]),(plus-minus)/(2*eps),delta=.004)

    def test_predict_ignores_specimen_id_and_supports_only_valid_positions(self):
        records = self.records()
        net = CNN({**self.c,'channels':3})
        model = dict(format_version=FORMAT, network_channels=3, point_order=POINTS, config=self.c,
                     input_contract=contract(self.c),model_version='test',
                     weights={k:v.tolist() for k,v in net.p.items()})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.json'; dump(path,model)
            result=predict(path,records)
            np.testing.assert_array_equal(result['scores'],predict(path,records[::-1])['scores'])
            renamed=copy.deepcopy(records)
            for r in renamed:
                r['stick_id']='UNKNOWN'; r['sample_id']=build_sample_id('UNKNOWN',1,0,r['impact_point_id'],1)
            np.testing.assert_array_equal(result['scores'],predict(path,renamed)['scores'])
            with self.assertRaises(ValueError): predict(path,[records[0],records[0],records[2]])
            with self.assertRaises(ValueError): predict(path,[records[1],records[0],records[2]])

    def test_export_resources_and_both_simple_baselines_are_reversible(self):
        time=np.random.default_rng(8).normal(size=(2,2400,3)).astype(np.float32)
        for baseline in ['amplitude_only','shape_without_amplitude']:
            np.testing.assert_array_equal(features(time,baseline),features(time[:,:,::-1],baseline))
        for mode,params in [('time',2514),('spectrum',12114)]:
            c={**self.c,'input_mode':mode}; net=CNN({**c,'channels':3})
            model=dict(config=c,input_contract=contract(c),weights={k:v.tolist() for k,v in net.p.items()})
            with tempfile.TemporaryDirectory() as folder:
                out=Path(folder); export_triplet(model,out)
                budget=json.loads((out/'deployment_budget.json').read_text(encoding='utf-8'))
                self.assertEqual(budget['parameters'],params)
                self.assertEqual(budget['input_channels'],3)
                self.assertEqual(budget['raw_input_bytes'],14400)
                header=(out/'model_weights.h').read_text(encoding='utf-8')
                self.assertIn('preprocess_three',header)
                self.assertIn('infer_either_direction',header)
                self.assertIn('reverse_input[i*3+p]=input[i*3+2-p]',header)


if __name__ == '__main__': unittest.main()
