import copy
import unittest
import numpy as np
from cnn_analysis.core import config,validate_config
from cnn_analysis.spectrum_preparation import VERSION
from cnn_analysis.triplet import inputs,contract
from cnn_analysis.transfer_remedies import transform


class AdoptedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.prep={'version':VERSION,'rms_target':[.1,.07,.1],
                   'log_reference':[.001,.002,.001],'log_gain':[.002,.001,.002]}
        self.c={**config(),'input_mode':'spectrum','spectrum_preparation':self.prep}

    def test_matches_research_preparation_exactly_and_reverses(self):
        x=np.random.default_rng(19).normal(size=(4,2400,3)).astype(np.float32)*.15
        original={k:v for k,v in self.c.items() if k!='spectrum_preparation'}
        expected=transform(x,original,self.prep,'rms_log')
        np.testing.assert_array_equal(inputs(x,self.c),expected)
        np.testing.assert_array_equal(inputs(x[:,:,::-1],self.c),expected[:,:,::-1])
        scaled=inputs(x*np.array([.5,2,1.5],dtype=np.float32)[None,None,:],self.c)
        np.testing.assert_allclose(scaled,expected,atol=1e-8,rtol=2e-5)
        self.assertEqual(contract(self.c)['version'],'WOOD_TRIPLET_RMS_LOG_V1')

    def test_invalid_contracts_rejected_and_zero_finite(self):
        for key,value in [('rms_target',[0,.1,0]),('log_reference',[.1,.1,.2]),('log_gain',[float('nan'),1,1])]:
            c=copy.deepcopy(self.c);c['spectrum_preparation'][key]=value
            with self.assertRaises(ValueError):validate_config(c)
        c={**self.c,'input_mode':'time'}
        with self.assertRaises(ValueError):validate_config(c)
        np.testing.assert_array_equal(inputs(np.zeros((1,2400,3),dtype=np.float32),self.c),np.zeros((1,1201,3),dtype=np.float32))


if __name__=='__main__':unittest.main()
