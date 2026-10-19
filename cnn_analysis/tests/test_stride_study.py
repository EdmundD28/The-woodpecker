import unittest
import numpy as np
from cnn_analysis.core import config
from cnn_analysis.network import CNN
from cnn_analysis.stride_study import StudyCNN, resources


class StrideStudyTests(unittest.TestCase):
    def test_baseline_matches_production_forward_and_gradients(self):
        c={**config(),'channels':3,'input_mode':'spectrum'}
        x=np.random.default_rng(9).normal(size=(2,1201,3)).astype(np.float32)*.1
        y=np.array([0,1]); a=CNN(c); b=StudyCNN(c)
        np.testing.assert_array_equal(a.forward(x),b.forward(x))
        la,ga=a.gradients(x,y); lb,gb=b.gradients(x,y)
        self.assertEqual(la,lb)
        for k in ga: np.testing.assert_array_equal(ga[k],gb[k])

    def test_band_and_native_head_gradients(self):
        for bands,strides in [(None,[2,2,1]),(None,[1,1,1]),(5,[1,1,1])]:
            c={**config(),'sample_count':48,'channels':3,'input_mode':'spectrum',
               'conv_layers':[[co,k,s] for co,k,s in zip([8,16,32],[9,5,3],strides)],'study_bands':bands}
            net=StudyCNN(c); rng=np.random.default_rng(11)
            x=rng.normal(size=(2,25,3)).astype(np.float32)*.2; y=np.array([0,1])
            _,g=net.gradients(x,y)
            for k,idx in [('w0',(4,1,2)),('w1',(2,2,3)),('w2',(1,3,5)),('wd',(12,1))]:
                original=net.p[k][idx].copy(); eps=.001
                net.p[k][idx]=original+eps; hi=net.gradients(x,y)[0]
                net.p[k][idx]=original-eps; lo=net.gradients(x,y)[0]
                net.p[k][idx]=original
                np.testing.assert_allclose(g[k][idx],(hi-lo)/(2*eps),atol=1e-4,rtol=.05)

    def test_matched_parameters_initialization_and_serialization(self):
        c={**config(),'channels':3,'input_mode':'spectrum'}; a=StudyCNN(c)
        cc={**c,'conv_layers':[[8,9,1],[16,5,1],[32,3,1]],'study_bands':151}
        b=StudyCNN(cc)
        self.assertEqual(resources(c,a)['parameters'],12114)
        self.assertEqual(resources(cc,b)['parameters'],12114)
        for k in a.p: np.testing.assert_array_equal(a.p[k],b.p[k])
        x=np.random.default_rng(4).normal(size=(2,1201,3)).astype(np.float32)
        np.testing.assert_array_equal(b.forward(x),StudyCNN(cc,b.p).forward(x))


if __name__=='__main__': unittest.main()
