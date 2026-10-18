"""Independently audit saved split manifests, frozen models and reported metrics."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .core import preprocess
from .cross_surface import quality_model, decision, dump
from .network import CNN
from .quality import EXCLUDED
from .surface_audit import surface
from .triplet import inputs, probabilities, scores


def run(out):
    out=Path(out); summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    records={}
    for src in summary['protocol']['sources']:
        payload=Path(src['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==src['sha256']
        records.update({r['sample_id']:r for r in map(json.loads,payload.decode('utf-8-sig').splitlines())})
    counts={'experiments':0,'cnn_models_reloaded':0,'triplets_audited':0,'cnn_probabilities_recomputed':0,'quality_models_rebuilt':0,'quality_decisions_recomputed':0}
    for exp in summary['experiments']:
        dest=out/f"{exp['source']}_to_{exp['target']}_seed{exp['seed']}"
        parts={}
        for kind in ['identity','quality']:
            pp={}
            with (dest/(kind+'_split.csv')).open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    sid=row['sample_id']; r=records[sid]
                    assert sid not in EXCLUDED
                    assert surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==exp['source']==row['surface']
                    pp.setdefault(row['partition'],[]).append(r)
            idsets=[{r['sample_id'] for r in rr} for rr in pp.values()]
            assert all(not(a&b) for a,b in itertools.combinations(idsets,2))
            assert len(set.union(*idsets))==summary['protocol']['normal_counts'][exp['source']]
            parts[kind]=pp
        for profile,nfeatures,factor in [('strict27',27,1),('shape25',25,1),('venue25',25,1.5)]:
            saved=json.loads((dest/(profile+'_model.json')).read_text(encoding='utf-8'))
            rebuilt=quality_model(parts['quality']['train'],parts['quality']['calibration'],saved['config'],nfeatures)
            assert saved==rebuilt, 'stored reference/scale/threshold must be reproducible from source fit/calibration alone'
            counts['quality_models_rebuilt']+=1
            for test in ['same_surface_normal','unseen_surface_normal','same_surface_abnormal','unseen_surface_abnormal']:
                rows=json.loads((dest/(profile+'_'+test+'.json')).read_text(encoding='utf-8'))
                for p in rows:
                    r=records[p['sample_id']]; actual=decision(r,saved,factor)
                    assert actual['retest']==p['retest'] and actual['score']==p['score'] and actual['threshold']==p['threshold']
                    expected_surface=exp['target'] if test.startswith('unseen') else exp['source']
                    assert surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==expected_surface
                    assert (r['sample_id'] in EXCLUDED)==test.endswith('_abnormal')
                    counts['quality_decisions_recomputed']+=1
        train_ids={r['sample_id'] for r in parts['identity']['train']}
        val_ids={r['sample_id'] for r in parts['identity']['validation']}
        same_ids={r['sample_id'] for r in parts['identity']['test']}
        pairings=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        for test,triples in pairings.items():
            seen=[]
            for ids in triples:
                rr=[records[sid] for sid in ids]
                assert [r['impact_point_id'] for r in rr]==[1,2,3]
                assert len({r['stick_id'] for r in rr})==1 and not(set(ids)&EXCLUDED)
                expected=exp['target'] if test.startswith('unseen') else exp['source']
                assert {surface(r['stick_id'],r['impact_point_id'],r['strike_id']) for r in rr}=={expected}
                pool=train_ids if test=='train' else val_ids if test=='validation' else same_ids if test.startswith('same') else set(records)-train_ids-val_ids-same_ids-EXCLUDED
                assert set(ids)<=pool
                seen.extend(ids);counts['triplets_audited']+=1
            if test.endswith('disjoint'):assert len(seen)==len(set(seen))
        for mode in ['time','spectrum']:
            md=dest/mode;c=json.loads((md/'config.json').read_text(encoding='utf-8'))
            with np.load(md/'weights.npz') as archive:net=CNN(c,dict(archive))
            counts['cnn_models_reloaded']+=1
            for test in ['validation','same_surface_test','unseen_surface_test','same_surface_disjoint','unseen_surface_disjoint']:
                rows=json.loads((md/(test+'_predictions.json')).read_text(encoding='utf-8'))
                assert [p['sample_ids'] for p in rows]==pairings[test]
                yy=np.array([int(records[p['sample_ids'][0]]['stick_id']=='WOOD02') for p in rows])
                assert yy.tolist()==[p['actual'] for p in rows]
                pp=np.asarray([p['probabilities'] for p in rows],dtype=np.float32)
                assert pp.argmax(1).tolist()==[p['prediction'] for p in rows]
                recomputed=scores(yy,pp);stored=exp['cnn'][mode][test]
                for key in ['count','accuracy','balanced_accuracy','per_class_recall','confusion_matrix_rows_actual_columns_predicted']:
                    assert recomputed[key]==stored[key]
                if test=='unseen_surface_test':
                    probes=np.linspace(0,len(rows)-1,16,dtype=int)
                    xx=np.stack([np.concatenate([preprocess(records[sid],c) for sid in rows[i]['sample_ids']],axis=1) for i in probes])
                    xx=inputs(xx,c);pr=probabilities(net,xx)
                    np.testing.assert_allclose(pr,pp[probes],atol=2e-6,rtol=2e-6)
                    np.testing.assert_array_equal(pr,probabilities(net,xx[:,:,::-1]))
                    counts['cnn_probabilities_recomputed']+=len(probes)
        counts['experiments']+=1
    hashes={}
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ['verification.json','artifact_sha256.json']:
            hashes[str(p.relative_to(out))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out/'artifact_sha256.json',hashes)
    result={'passed':True,'counts':counts,'raw_file_hashes_unchanged':True,'source_only_quality_reconstruction':True,'saved_cnn_predictions_and_direction_invariance':True,'artifact_hash_count':len(hashes)}
    dump(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
