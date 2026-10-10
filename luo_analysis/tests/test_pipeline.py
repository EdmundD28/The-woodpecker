import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from luo_analysis.core import config, preprocess, load, split, model_input, input_contract
from luo_analysis.network import CNN, windows, fit_centroid, centroid_predict
from luo_analysis.__main__ import demo_data, predict, train, train_candidates
from luo_analysis.export import export


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.c=config()
        p=Path(__file__).resolve().parents[2]/'pc_collector'/'examples'/'WOOD_IMPACT_V1_example.jsonl'
        self.r=json.loads(p.read_text().splitlines()[0])

    def test_exact_preprocessing_firmware_contract(self):
        x=preprocess(self.r,self.c)
        raw=np.array(self.r['raw'])
        expected=((raw-sum(self.r['raw'])/1024)/2048).astype(np.float32).reshape(1024,1)
        np.testing.assert_array_equal(x,expected)
        self.assertEqual(x.dtype,np.float32)
        self.assertEqual(x.nbytes,4096)

    def test_adc_offset_invariance(self):
        shifted=copy.deepcopy(self.r)
        shifted['raw']=[v+30 for v in shifted['raw']]
        np.testing.assert_array_equal(preprocess(self.r,self.c),preprocess(shifted,self.c))

    def test_reject_quality_and_protocol(self):
        cases=[]
        for key,value in [('sample_rate_hz',8000),('sample_count',1023),('anomaly_flags',['WEAK_SIGNAL']),('raw',[2048]*1024)]:
            r=copy.deepcopy(self.r); r[key]=value; cases.append(r)
        for value in [-1,4096,0,4095,True,float('nan')]:
            r=copy.deepcopy(self.r); r['raw'][0]=value; cases.append(r)
        r=copy.deepcopy(self.r); r['raw'].pop(); cases.append(r)
        r=copy.deepcopy(self.r); r.pop('sample_id'); cases.append(r)
        for r in cases:
            with self.subTest(r=str(r)[:40]),self.assertRaises((ValueError,TypeError)):
                preprocess(r,self.c)

    def test_duplicate_json_and_jsonl_count_once(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'one.json').write_text(json.dumps(self.r))
            (p/'many.jsonl').write_text(json.dumps(self.r)+'\n')
            a,r,d=load([p],self.c)
            self.assertEqual((len(a),len(r),len(d)),(1,0,1))
            self.assertEqual(a[0]['source'],'unknown')
            self.assertIsNone(a[0]['label'])

    def test_conflicting_duplicate_is_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            r=copy.deepcopy(self.r); r['raw'][0]+=1
            p=Path(folder)/'data.jsonl'; p.write_text(json.dumps(self.r)+'\n'+json.dumps(r))
            with self.assertRaises(RuntimeError): load([p],self.c)

    def test_malformed_rows_logged(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'data.jsonl'; p.write_text('not json\n'+json.dumps(self.r))
            a,r,d=load([p],self.c)
            self.assertEqual((len(a),len(r)),(1,1))

    def test_group_isolation_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            data,manifest=demo_data(self.c,Path(folder))
            a,_,_=load([data],self.c,manifest)
            ids,groups=split(a,self.c)
            self.assertEqual(groups,split(a,self.c)[1])
            for one,two in [('train','test'),('train','validation'),('validation','test')]:
                self.assertFalse(set(groups[one])&set(groups[two]))
            self.assertEqual(sum(len(ix) for ix in ids.values()),len(a))
            # Same E/R numbers are kept together even across sticks/positions.
            for group in set(r['group'] for r in a):
                self.assertEqual(sum(group in gs for gs in groups.values()),1)

    def test_insufficient_batches_refused(self):
        with self.assertRaises(ValueError): split([],self.c)

    def test_empty_real_data_exports_no_model(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            train([],self.c,out)
            self.assertEqual(json.loads((out/'training_status.json').read_text())['status'],'PENDING')
            self.assertFalse((out/'model.json').exists())

    def test_unknown_provenance_cannot_train(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            train([{'label':'solid','source':'unknown'}],self.c,out)
            self.assertEqual(json.loads((out/'training_status.json').read_text())['status'],'PENDING')

    def test_window_same_padding(self):
        x=np.arange(8,dtype=np.float32).reshape(1,8,1)
        view,left=windows(x,3,2)
        self.assertEqual(left,0)
        np.testing.assert_array_equal(view[0,-1,:,0],[6,7,0])

    def test_spectrum_contract_matches_direct_dft(self):
        c={**self.c,'input_mode':'spectrum'}
        t=preprocess(self.r,c)[None,:,:]
        spectrum=model_input(t,c)
        self.assertEqual(spectrum.shape,(1,513,1))
        self.assertEqual(spectrum.dtype,np.float32)
        windowed=t[0,:,0].astype(float)*np.hanning(1024)
        for k in [0,1,53,256,512]:
            at=np.arange(1024)
            re=np.sum(windowed*np.cos(2*np.pi*k*at/1024))
            im=-np.sum(windowed*np.sin(2*np.pi*k*at/1024))
            self.assertAlmostEqual(float(spectrum[0,k,0]),np.hypot(re,im)/1024,delta=1e-8)
        self.assertEqual(input_contract(c)['version'],'SPECTRUM_V0_1')
        self.assertEqual(CNN(c).forward(spectrum).shape,(1,2))

    def test_both_pending_candidates_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); train_candidates([],self.c,p)
            comparison=json.loads((p/'candidate_comparison.json').read_text())
            self.assertEqual(set(comparison['candidates']),{'time','spectrum'})
            self.assertEqual(comparison['status'],'PENDING')

    def test_spectrum_predict_and_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); c={**self.c,'input_mode':'spectrum'}; net=CNN(c)
            m={'format_version':'WOOD_CNN_NUMPY_V1','model_version':'DEMO_ONLY-spectrum','demo_only':True,
               'config':c,'input_contract':input_contract(c),'weights':{k:v.tolist() for k,v in net.p.items()}}
            (p/'model.json').write_text(json.dumps(m)); (p/'record.json').write_text(json.dumps(self.r))
            result=predict(p/'model.json',p/'record.json')
            expected=net.forward(model_input(preprocess(self.r,c)[None,:,:],c))[0]
            np.testing.assert_array_equal(result['scores'],expected)
            export(m,p)
            budget=json.loads((p/'deployment_budget.json').read_text())
            self.assertEqual(budget['input_bytes'],513*4)
            header=(p/'model_weights.h').read_text()
            self.assertIn('namespace wood_model_spectrum',header)
            self.assertIn('conv(input,513,1',header)
            self.assertIn('for(int c=0;c<2080;++c) scores',header)
            m['input_contract']=input_contract(self.c)
            (p/'model.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): predict(p/'model.json',p/'record.json')

    def test_cnn_forward_matches_scalar_convolution_reference(self):
        net=CNN(self.c)
        x=np.random.default_rng(1).normal(size=(1,32,1)).astype(np.float32)
        ref=x.copy()
        for i,(cout,k,s) in enumerate(self.c['conv_layers']):
            olen=(ref.shape[1]+s-1)//s
            left=max((olen-1)*s+k-ref.shape[1],0)//2
            z=np.zeros((1,olen,cout),dtype=np.float32)
            for t in range(olen):
                for o in range(cout):
                    value=np.float32(net.p[f'b{i}'][o])
                    for kk in range(k):
                        at=t*s+kk-left
                        if 0<=at<ref.shape[1]:
                            for cin in range(ref.shape[2]):
                                value+=ref[0,at,cin]*net.p[f'w{i}'][kk,cin,o]
                    z[0,t,o]=max(value,0)
            ref=z
        pooled=ref.mean(1)
        logits=pooled@net.p['wd']+net.p['bd']
        probs=np.exp(logits-logits.max(1,keepdims=True)); probs/=probs.sum(1,keepdims=True)
        np.testing.assert_allclose(net.forward(x),probs,rtol=1e-5,atol=1e-6)

    def test_cnn_gradients_numerically(self):
        net=CNN(self.c)
        x=np.random.default_rng(7).normal(size=(2,32,1)).astype(np.float32); y=np.array([0,1])
        _,g=net.gradients(x,y)
        eps=.001
        for name in ['w0','w1','w2','wd','bd']:
            idx=np.unravel_index(np.abs(g[name]).argmax(),g[name].shape)
            original=net.p[name][idx].copy()
            net.p[name][idx]=original+eps; plus=net.gradients(x,y)[0]
            net.p[name][idx]=original-eps; minus=net.gradients(x,y)[0]
            net.p[name][idx]=original
            self.assertAlmostEqual(float(g[name][idx]),(plus-minus)/(2*eps),delta=.004,msg=name)

    def test_weight_shape_checked(self):
        net=CNN(self.c); weights={k:v.tolist() for k,v in net.p.items()}; weights['wd']=[1]
        with self.assertRaises(ValueError): CNN(self.c,weights)

    def test_centroid_fits_training_only(self):
        x=np.array([[0.,1.],[1.,1.],[8.,2.],[9.,2.]])
        y=np.array([0,0,1,1]); m=fit_centroid(x,y)
        np.testing.assert_allclose(m['mean'],[4.5,1.5])
        np.testing.assert_array_equal(centroid_predict(x,m),y)

    def test_model_roundtrip_and_demo_output(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); net=CNN(self.c)
            m={'format_version':'WOOD_CNN_NUMPY_V1','model_version':'DEMO_ONLY-test','demo_only':True,
               'config':self.c,'weights':{k:v.tolist() for k,v in net.p.items()}}
            (p/'model.json').write_text(json.dumps(m)); (p/'record.json').write_text(json.dumps(self.r))
            result=predict(p/'model.json',p/'record.json')
            self.assertEqual(result['status'],'DEMO_ONLY')
            expected=net.forward(preprocess(self.r,self.c)[None,:,:])[0]
            np.testing.assert_array_equal(result['scores'],expected)
            r=copy.deepcopy(self.r); r['anomaly_flags']=['CLIPPED']
            (p/'record.json').write_text(json.dumps(r))
            self.assertEqual(predict(p/'model.json',p/'record.json')['status'],'INVALID_SIGNAL')
            export(m,p)
            self.assertTrue((p/'model_weights.h').exists())

    def test_output_tie_and_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); net=CNN(self.c)
            for a in net.p.values(): a.fill(0)
            m={'format_version':'WOOD_CNN_NUMPY_V1','model_version':'test','demo_only':False,
               'config':self.c,'weights':{k:v.tolist() for k,v in net.p.items()}}
            (p/'model.json').write_text(json.dumps(m)); (p/'record.json').write_text(json.dumps(self.r))
            self.assertEqual(predict(p/'model.json',p/'record.json')['reason'],'TIED_TOP_SCORES')
            net.p['bd'][0]=1
            m['weights']={k:v.tolist() for k,v in net.p.items()}; m['config']['confidence_threshold']=.99
            (p/'model.json').write_text(json.dumps(m))
            self.assertEqual(predict(p/'model.json',p/'record.json')['status'],'LOW_CONFIDENCE')


if __name__=='__main__': unittest.main()
