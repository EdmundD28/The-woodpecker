"""Reconstruct split membership and independently check frozen ablation predictions."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from .core import preprocess
from .cross_surface import assemble, dump
from .quality import EXCLUDED
from .stride_study import StudyCNN, VARIANTS
from .surface_audit import surface
from .triplet import inputs, probabilities, scores


def verify(out):
    out=Path(out); summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    base=Path(summary['protocol']['base_run'])
    assert hashlib.sha256((base/'summary.json').read_bytes()).hexdigest()==summary['protocol']['base_summary_sha256']
    old=json.loads((base/'summary.json').read_text(encoding='utf-8'))
    records={}
    for src in old['protocol']['sources']:
        payload=Path(src['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==src['sha256']
        records.update({r['sample_id']:r for r in map(json.loads,payload.decode('utf-8-sig').splitlines())})
    frozen=json.loads((out/'all_models_frozen.json').read_text(encoding='utf-8'))['models']
    assert len(frozen)==len(summary['experiments'])==24
    counts={'models':0,'metrics_recomputed':0,'probabilities_recomputed':0,'split_manifests_checked':0}
    for exp in old['experiments']:
        tag=f"{exp['source']}_to_{exp['target']}_seed{exp['seed']}"; dest=out/tag
        for fn in ['identity_split.csv','triplet_sample_ids.json']:
            assert (dest/fn).read_bytes()==(base/tag/fn).read_bytes()
        parts={}
        with (dest/'identity_split.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                sid=row['sample_id']; r=records[sid]
                assert sid not in EXCLUDED
                assert surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==exp['source']
                parts.setdefault(row['partition'],set()).add(sid)
        assert all(not(a&b) for a,b in itertools.combinations(parts.values(),2))
        pairs=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        for test,triples in pairs.items():
            used=[]
            for ids in triples:
                rr=[records[sid] for sid in ids]
                assert [r['impact_point_id'] for r in rr]==[1,2,3]
                assert len({r['stick_id'] for r in rr})==1 and not(set(ids)&EXCLUDED)
                expected=exp['target'] if test.startswith('unseen') else exp['source']
                assert {surface(r['stick_id'],r['impact_point_id'],r['strike_id']) for r in rr}=={expected}
                if not test.startswith('unseen'):
                    pool=parts[test] if test in parts else parts['test']
                    assert set(ids)<=pool
                else: assert not(set(ids)&set.union(*parts.values()))
                used.extend(ids)
            if test.endswith('disjoint'): assert len(used)==len(set(used))
        counts['split_manifests_checked']+=1
        losses={}
        for variant in VARIANTS:
            md=dest/variant; c=json.loads((md/'config.json').read_text(encoding='utf-8'))
            info=json.loads((md/'evaluation.json').read_text(encoding='utf-8'))
            record=next(x for x in frozen if x['source']==exp['source'] and x['seed']==exp['seed'] and x['variant']==variant)
            assert record=={k:info[k] for k in record}
            assert hashlib.sha256((md/'weights.npz').read_bytes()).hexdigest()==record['weights_sha256']
            history=json.loads((md/'history.json').read_text(encoding='utf-8'))
            losses[variant]=min(r['validation_loss'] for r in history)
            assert losses[variant]==info['selection_validation_loss']
            assert min(history,key=lambda r:r['validation_loss'])['epoch']==info['best_epoch']
            with np.load(md/'weights.npz') as w: net=StudyCNN(c,dict(w))
            cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
            for test in ['validation','same_surface_test','unseen_surface_test','same_surface_disjoint','unseen_surface_disjoint']:
                with np.load(md/(test+'_predictions.npz')) as z: pp=z['probabilities']; yy=z['labels']
                expected=np.array([int(records[ids[0]]['stick_id']=='WOOD02') for ids in pairs[test]])
                np.testing.assert_array_equal(yy,expected)
                assert scores(yy,pp)==info['evaluation'][test]
                counts['metrics_recomputed']+=1
                # Check every disjoint triplet plus 16 fixed probes from each larger partition.
                probes=np.arange(len(yy)) if test.endswith('disjoint') else np.linspace(0,len(yy)-1,min(16,len(yy)),dtype=int)
                xx,labels=assemble([pairs[test][int(i)] for i in probes],records,cache)
                xx=inputs(xx,c); pr=probabilities(net,xx,batch_size=16)
                np.testing.assert_allclose(pr,pp[probes],atol=2e-6,rtol=2e-6)
                np.testing.assert_array_equal(pr,probabilities(net,xx[:,:,::-1],batch_size=16))
                counts['probabilities_recomputed']+=len(probes)
            if variant=='stride222':
                ref=json.loads((base/tag/'spectrum/evaluation.json').read_text(encoding='utf-8'))
                for test in ['same_surface_test','unseen_surface_test','unseen_surface_disjoint']:
                    assert info['previous_run_baseline_delta'][test]==info['evaluation'][test]['balanced_accuracy']-ref[test]['balanced_accuracy']
            counts['models']+=1
        sel=json.loads((dest/'selection_locked.json').read_text(encoding='utf-8'))
        assert losses==sel['losses'] and min(losses,key=losses.get)==sel['selected']
    result={'passed':True,'counts':counts,'previous_baseline_deltas_audited':True,
            'exact_previous_pairings_and_source_only_selection':True,'frozen_weight_hashes_unchanged':True}
    dump(out/'verification.json',result)
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
    dump(out/'artifact_sha256.json',hashes)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);verify(p.parse_args().run)
