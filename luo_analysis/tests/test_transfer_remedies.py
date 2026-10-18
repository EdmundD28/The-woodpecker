import unittest
import numpy as np
from luo_analysis.core import config
from luo_analysis.network import CNN
from luo_analysis.transfer_remedies import transform, RemedyCNN, band_features, METHODS
from luo_analysis.triplet import inputs


class TransferRemedyTests(unittest.TestCase):
    def setUp(self):
        self.c={**config(),'channels':3,'input_mode':'spectrum'}
        self.x=np.random.default_rng(21).normal(size=(4,2400,3)).astype(np.float32)*.1
        self.prep={'rms_target':[.1,.07,.1],'spectral_l1_target':[.8,.5,.8],
                   'log_reference':[.001,.002,.001],'log_gain':[.001,.002,.001]}

    def test_raw_and_gradient_match_production(self):
        x=transform(self.x,self.c,self.prep,'raw');y=np.array([0,1,0,1])
        np.testing.assert_array_equal(x,inputs(self.x,self.c))
        a=CNN(self.c);b=RemedyCNN(self.c,'raw')
        np.testing.assert_array_equal(a.forward(x),b.forward(x))
        la,ga=a.gradients(x,y);lb,gb=b.gradients(x,y)
        self.assertEqual(la,lb)
        for k in ga:np.testing.assert_array_equal(ga[k],gb[k])

    def test_normalization_gain_and_endpoint_invariance(self):
        gains=np.array([.5,2,1.5],dtype=np.float32)[None,None,:]
        for method in METHODS:
            a=transform(self.x,self.c,self.prep,method)
            np.testing.assert_allclose(a[:,:,::-1],transform(self.x[:,:,::-1],self.c,self.prep,method),atol=1e-8,rtol=1e-6)
            if method not in ['raw','gain_aug']:
                np.testing.assert_allclose(a,transform(self.x*gains,self.c,self.prep,method),atol=1e-8,rtol=2e-5)
        for bands in [16,64]:
            a=band_features(self.x,self.c,bands)
            np.testing.assert_allclose(a,band_features(self.x*gains,self.c,bands),atol=1e-8)
            np.testing.assert_array_equal(a,band_features(self.x[:,:,::-1],self.c,bands))

    def test_augmentation_reproducible_and_shape_energy_preserved(self):
        x=inputs(self.x,self.c)
        for method in ['gain_aug','rms_shape_aug']:
            a=RemedyCNN(self.c,method);b=RemedyCNN(self.c,method)
            transformed=a.augment(x)
            np.testing.assert_array_equal(transformed,b.augment(x))
            self.assertFalse(np.array_equal(transformed,a.augment(x)))
            self.assertTrue(np.isfinite(transformed).all())
            if method=='rms_shape_aug':
                np.testing.assert_allclose(np.sum(transformed.astype(np.float64)**2,1),np.sum(x.astype(np.float64)**2,1),rtol=1e-6,atol=1e-12)


if __name__=='__main__':unittest.main()
