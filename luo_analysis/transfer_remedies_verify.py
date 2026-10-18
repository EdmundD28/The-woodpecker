"""Audit source-only preparation, frozen remedy models, metrics and predictions."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from .core import preprocess
from .cross_surface import assemble,dump
from .network import centroid_predict
from .quality import EXCLUDED
from .surface_audit import surface
from .transfer_remedies import setup,fit_preparation,transform,RemedyCNN,band_features,METHODS,CLASSICAL
from .triplet import probabilities,scores


def verify(out):
    out=Path(out);summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    base=Path(summary['protocol']['base_run']);old,records,c=setup(base,out)
    assert hashlib.sha256((base/'summary.json').read_bytes()).hexdigest()==summary['protocol']['base_summary_sha256']
    frozen=json.loads((out/'all_models_frozen.json').read_text(encoding='utf-8'))['models']
    assert len(frozen)==len(summary['experiments'])==48
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    counts={'cnn_models':0,'centroid_models':0,'preparations_rebuilt':0,'metrics_recomputed':0,'probabilities_recomputed':0,'sensitivity_cases_recomputed':0}
    for exp in old['experiments']:
        tag=f"{exp['source']}_to_{exp['target']}_seed{exp['seed']}";dest=out/tag
        for fn in ['identity_split.csv','triplet_sample_ids.json']:
            assert (dest/fn).read_bytes()==(base/tag/fn).read_bytes()
        parts={}
        with (dest/'identity_split.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                sid=row['sample_id'];r=records[sid];assert sid not in EXCLUDED
                assert surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==exp['source']
                parts.setdefault(row['partition'],set()).add(sid)
        assert all(not(a&b) for a,b in itertools.combinations(parts.values(),2))
        cc=json.loads((dest/'raw/config.json').read_text(encoding='utf-8'))
        prep=json.loads((dest/'preparation.json').read_text(encoding='utf-8'))
        assert prep==fit_preparation(records,parts['train'],cc)
        counts['preparations_rebuilt']+=1
        pairs=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        for test,triples in pairs.items():
            flat=[]
            for ids in triples:
                rr=[records[sid] for sid in ids]
                assert [r['impact_point_id'] for r in rr]==[1,2,3]
                assert len({r['stick_id'] for r in rr})==1 and not(set(ids)&EXCLUDED)
                expected=exp['target'] if test.startswith('unseen') else exp['source']
                assert {surface(r['stick_id'],r['impact_point_id'],r['strike_id']) for r in rr}=={expected}
                if test.startswith('unseen'):assert not(set(ids)&set.union(*parts.values()))
                else:assert set(ids)<=(parts[test] if test in parts else parts['test'])
                flat.extend(ids)
            if test.endswith('disjoint'):assert len(flat)==len(set(flat))
        losses={};classical_scores={}
        for method in METHODS+CLASSICAL:
            md=dest/method;info=json.loads((md/'evaluation.json').read_text(encoding='utf-8'))
            ff=next(x for x in frozen if x['source']==exp['source'] and x['seed']==exp['seed'] and x['method']==method)
            assert ff=={k:info[k] for k in ff}
            if method in METHODS:
                assert hashlib.sha256((md/'weights.npz').read_bytes()).hexdigest()==ff['weights_sha256']
                with np.load(md/'weights.npz') as w:net=RemedyCNN(cc,method,dict(w))
                def predict(x):return probabilities(net,transform(x,cc,prep,method))
                history=json.loads((md/'history.json').read_text(encoding='utf-8'))
                losses[method]=min(r['validation_loss'] for r in history)
                assert losses[method]==info['selection_loss']
                if method=='raw':
                    with np.load(base/tag/'spectrum/weights.npz') as w:
                        for k in w.files:np.testing.assert_array_equal(net.p[k],w[k])
                    assert history==json.loads((base/tag/'spectrum/history.json').read_text(encoding='utf-8'))
                counts['cnn_models']+=1
            else:
                assert hashlib.sha256((md/'centroid.json').read_bytes()).hexdigest()==ff['centroid_sha256']
                fm=json.loads((md/'centroid.json').read_text(encoding='utf-8'))
                def predict(x):return np.eye(2,dtype=np.float32)[centroid_predict(band_features(x,cc,info['bands']),fm)]
                # Refit from the exact source triplets, including the same training weighting.
                from .network import fit_centroid
                xt,yt=assemble(pairs['train'],records,cache)
                assert fm==fit_centroid(band_features(xt,cc,info['bands']),yt)
                classical_scores[method]=info['validation']['balanced_accuracy']
                counts['centroid_models']+=1
            for test in ['validation','same_surface_test','unseen_surface_test','same_surface_disjoint','unseen_surface_disjoint']:
                with np.load(md/(test+'_predictions.npz')) as a:pp=a['probabilities'];yy=a['labels']
                expected=np.array([int(records[ids[0]]['stick_id']=='WOOD02') for ids in pairs[test]])
                np.testing.assert_array_equal(yy,expected)
                assert scores(yy,pp)==info['evaluation'][test]
                counts['metrics_recomputed']+=1
                probes=np.arange(len(yy)) if test.endswith('disjoint') else np.linspace(0,len(yy)-1,min(16,len(yy)),dtype=int)
                xx,_=assemble([pairs[test][int(i)] for i in probes],records,cache)
                pr=predict(xx);np.testing.assert_allclose(pr,pp[probes],atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(pr,predict(xx[:,:,::-1]),atol=1e-6,rtol=1e-6)
                counts['probabilities_recomputed']+=len(probes)
                if test=='unseen_surface_disjoint':
                    for case in info['evaluation']['ideal_gain_sensitivity']:
                        changed=predict((xx*np.array(case['gains'])[None,None,:]).astype(np.float32))
                        assert int((changed.argmax(1)!=pr.argmax(1)).sum())==case['prediction_flips']
                        assert scores(yy,changed)['balanced_accuracy']==case['balanced_accuracy']
                        counts['sensitivity_cases_recomputed']+=1
                if method=='raw':
                    ref=json.loads((base/tag/'spectrum/evaluation.json').read_text(encoding='utf-8'))
                    if test in ['same_surface_test','unseen_surface_test','unseen_surface_disjoint']:
                        assert info['evaluation'][test]==ref[test]
        selection=json.loads((dest/'selection_locked.json').read_text(encoding='utf-8'))
        assert selection['cnn_loss']==losses and selection['cnn_selected']==min(losses,key=losses.get)
        assert selection['classical_selected']==max(CLASSICAL,key=lambda m:classical_scores[m])
    result={'passed':True,'counts':counts,'raw_baseline_exactly_reproduced':True,'source_only_preparation_rebuilt':True,
            'original_splits_and_exclusions_unchanged':True,'all_frozen_hashes_unchanged':True}
    dump(out/'verification.json',result)
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
    dump(out/'artifact_sha256.json',hashes);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);verify(p.parse_args().run)
