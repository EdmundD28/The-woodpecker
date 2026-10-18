"""Predeclared, source-only training and calibration for support-surface transfer."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from .core import config, preprocess
from .network import CNN, signal_features, fit_centroid, centroid_predict
from .quality import EXCLUDED, fit, calibrate, predict
from .surface_audit import surface
from .triplet import inputs, probabilities, scores, features

SEEDS = [42, 123, 2026]
SURFACES = ['carpet', 'table']


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def split_records(rows, seed, quality=False):
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ['train', 'calibration' if quality else 'validation', 'test']}
    for wood, point in itertools.product(['WOOD01', 'WOOD02'], [1, 2, 3]):
        pool = sorted([r for r in rows if r['stick_id'] == wood and r['impact_point_id'] == point], key=lambda r:r['sample_id'])
        order = rng.permutation(len(pool))
        ntrain = int(len(pool)*(.4 if quality else .6))
        nval = math.ceil(len(pool)*.4) if quality else max(1, int(len(pool)*.2))
        assert ntrain and nval and len(pool)-ntrain-nval > 0
        for name, ids in zip(parts, [order[:ntrain], order[ntrain:ntrain+nval], order[ntrain+nval:]]):
            parts[name].extend(pool[int(i)] for i in ids)
    sets = [{r['sample_id'] for r in part} for part in parts.values()]
    assert all(not (a & b) for a, b in itertools.combinations(sets, 2))
    assert set.union(*sets) == {r['sample_id'] for r in rows}
    return parts


def combinations(rows, count, seed, disjoint=False):
    """Build homogeneous-surface homogeneous-wood triples, exactly one hit per point."""
    rng = np.random.default_rng(seed)
    result = []
    for wood in ['WOOD01', 'WOOD02']:
        pools = [sorted([r['sample_id'] for r in rows if r['stick_id'] == wood and r['impact_point_id'] == p]) for p in [1, 2, 3]]
        if disjoint:
            pools = [rng.permutation(p).tolist() for p in pools]
            triples = list(zip(*pools))
        else:
            total = math.prod(map(len, pools))
            wanted = min(count, total)
            # Cycle through every hit first; balance classes only in metrics, not by duplicating rows.
            perm = [rng.permutation(p).tolist() for p in pools]
            triples = list(dict.fromkeys(tuple(p[i % len(p)] for p in perm) for i in range(max(map(len, perm)))))
            if len(triples) > wanted:
                triples = triples[:wanted]
            seen = set(triples)
            while len(triples) < wanted:
                row = tuple(p[int(rng.integers(len(p)))] for p in pools)
                if row not in seen:
                    seen.add(row); triples.append(row)
        result.extend(triples)
    assert len(result) == len(set(result))
    return result


def assemble(triples, records, cache):
    for ids in triples:
        rr = [records[sid] for sid in ids]
        assert [r['impact_point_id'] for r in rr] == [1, 2, 3]
        assert len({r['stick_id'] for r in rr}) == 1
        assert len({surface(r['stick_id'], r['impact_point_id'], r['strike_id']) for r in rr}) == 1
        assert not (set(ids) & EXCLUDED)
    return np.stack([np.stack([cache[sid] for sid in ids], axis=1) for ids in triples]), np.array([int(records[ids[0]]['stick_id'] == 'WOOD02') for ids in triples])


def quality_model(rows, calibration, c, feature_count):
    model = fit(rows, c)
    model['feature_names'] = model['feature_names'][:feature_count]
    for g in model['groups'].values():
        g['median'] = g['median'][:feature_count]
        g['scale'] = g['scale'][:feature_count]
        g['reference'] = [r[:feature_count] for r in g['reference']]
    return calibrate(model, calibration, .05)


def decision(record, model, factor):
    p = predict(record, model)
    if p['threshold'] is not None:
        p['threshold'] *= factor
        p['retest'] = p['score'] > p['threshold']
    return p


def quality_metrics(rows, model, factor):
    predictions = []
    groups = {}
    for r in rows:
        p = decision(r, model, factor)
        p.update(stick_id=r['stick_id'], point=r['impact_point_id'])
        predictions.append(p)
        for key in ['all', f"{r['stick_id']}:P{r['impact_point_id']}"]:
            g = groups.setdefault(key, {'count': 0, 'retest': 0})
            g['count'] += 1; g['retest'] += int(p['retest'])
    for g in groups.values():
        g['retest_rate'] = g['retest']/g['count']
    return groups, predictions


def baseline_metric(y, pred):
    return scores(y, np.eye(2)[pred])


def manifest(parts, path):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['sample_id', 'partition', 'surface', 'wood', 'point'])
        for name, rows in parts.items():
            for r in rows:
                w.writerow([r['sample_id'], name, surface(r['stick_id'], r['impact_point_id'], r['strike_id']), r['stick_id'], r['impact_point_id']])


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    c = config(); c.update(classes=['WOOD01','WOOD02'], class_display_names=['木头1','木头2'])
    audit = Path(__file__).parent/'runs/20261019_surface_labels_audit/summary.json'
    sources = json.loads(audit.read_text(encoding='utf-8'))['sources']
    records = {}
    raw_hashes = set()
    for src in sources:
        payload = Path(src['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == src['sha256']
        rows = [json.loads(s) for s in payload.decode('utf-8-sig').splitlines() if s.strip()]
        assert [r['strike_id'] for r in rows] == list(range(1, 51))
        for r in rows:
            preprocess(r, c)
            digest = hashlib.sha256(np.asarray(r['raw'], dtype='<u2').tobytes()).hexdigest()
            assert r['sample_id'] not in records and digest not in raw_hashes
            records[r['sample_id']] = r; raw_hashes.add(digest)
    assert len(records) == 300 and EXCLUDED <= set(records)
    normals = {s:[r for r in records.values() if r['sample_id'] not in EXCLUDED and surface(r['stick_id'],r['impact_point_id'],r['strike_id']) == s] for s in SURFACES}
    abnormal = {s:[r for r in records.values() if r['sample_id'] in EXCLUDED and surface(r['stick_id'],r['impact_point_id'],r['strike_id']) == s] for s in SURFACES}
    cache = {sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    protocol = {'sources':sources, 'seeds':SEEDS, 'excluded':sorted(EXCLUDED), 'normal_counts':{s:len(rows) for s,rows in normals.items()},
        'identity':{'split':'source only, 60/20/remainder per wood/point', 'train_combinations_per_wood':120, 'validation_combinations_max_per_wood':80,
                    'test_combinations_max_per_wood':1000, 'target_pairing_seed':191019, 'epochs':120,'patience':20,'learning_rate':c['learning_rate'],
                    'modes':['time','spectrum'], 'direction':'mean forward and endpoint reversed probabilities, training reversal augmentation',
                    'selection':'source validation loss only; report all seeds and modes; never select on target'},
        'quality':{'split':'source only, floor(40%) fit / ceil(40%) calibration / remainder test, per wood/point',
                   'reason':'at least 19 middle calibration normals needed for finite 5% threshold', 'k':5,'alpha':.05,
                   'profiles':{'strict27':{'feature_count':27,'factor':1}, 'shape25':{'feature_count':25,'factor':1}, 'venue25':{'feature_count':25,'factor':1.5}}},
        'baselines':['triplet amplitude-only centroid','triplet shape-only centroid','single-point shape centroid','single-point amplitude centroid'],
        'constraints':['no target-surface calibration or model selection','no mixed surfaces within a triple','no abnormal fitting or calibration',
                       'combination accuracy is not independent sample accuracy','same two specimens, one session per point, surface confounded with collection order',
                       'five abnormalities were inspected previously; remaining 295 presumed normal','no deployed model or firmware replacement']}
    dump(out/'protocol_locked.json', protocol)
    all_results = []
    start = time.monotonic()
    for source in SURFACES:
        target = next(s for s in SURFACES if s != source)
        for seed in SEEDS:
            tag = f'{source}_to_{target}_seed{seed}'
            dest = out/tag; dest.mkdir()
            print('START '+tag, flush=True)
            iparts = split_records(normals[source], seed)
            qparts = split_records(normals[source], seed, quality=True)
            manifest(iparts, dest/'identity_split.csv'); manifest(qparts, dest/'quality_split.csv')
            assert all(surface(r['stick_id'],r['impact_point_id'],r['strike_id']) == source for part in iparts.values() for r in part)
            qresults = {}; gates = {}
            for profile, nfeatures, factor in [('strict27',27,1),('shape25',25,1),('venue25',25,1.5)]:
                qm = quality_model(qparts['train'], qparts['calibration'], c, nfeatures)
                dump(dest/(profile+'_model.json'), qm)
                qr = {'thresholds':{k:{a:v[a] for a in ['threshold','calibration_count','threshold_rank']} for k,v in qm['groups'].items()}, 'factor':factor}
                for name, rows in [('same_surface_normal',qparts['test']),('unseen_surface_normal',normals[target]),('same_surface_abnormal',abnormal[source]),('unseen_surface_abnormal',abnormal[target])]:
                    qmtr, preds = quality_metrics(rows, qm, factor)
                    qr[name] = qmtr; dump(dest/(profile+'_'+name+'.json'), preds)
                    if name == 'unseen_surface_normal': gates[profile] = {p['sample_id']:not p['retest'] for p in preds}
                qresults[profile] = qr
            dump(dest/'quality_results.json', qresults)
            pairings = {
                'train':combinations(iparts['train'],120,seed), 'validation':combinations(iparts['validation'],80,seed+1),
                'same_surface_test':combinations(iparts['test'],1000,191019),
                'unseen_surface_test':combinations(normals[target],1000,191019),
                'same_surface_disjoint':combinations(iparts['test'],0,191019,True),
                'unseen_surface_disjoint':combinations(normals[target],0,191019,True)}
            dump(dest/'triplet_sample_ids.json', pairings)
            assembled = {name:assemble(ids,records,cache) for name,ids in pairings.items()}
            for name in ['same_surface_disjoint','unseen_surface_disjoint']:
                flat = list(itertools.chain.from_iterable(pairings[name])); assert len(flat)==len(set(flat))
            baselines = {}
            for mode in ['amplitude_only','shape_without_amplitude']:
                xx, yy = assembled['train']; fm = fit_centroid(features(xx,mode),yy)
                baselines[mode] = {name:baseline_metric(y,centroid_predict(features(x,mode),fm)) for name,(x,y) in assembled.items() if name not in ['train','validation']}
                dump(dest/(mode+'_centroid.json'),fm)
            single = {}
            for point in [1,2,3]:
                pools = {name:[r for r in rows if r['impact_point_id']==point] for name,rows in [('train',iparts['train']),('same_surface_test',iparts['test']),('unseen_surface_test',normals[target])]}
                for kind in ['shape','amplitude']:
                    def sf(rows):
                        x = np.stack([cache[r['sample_id']] for r in rows])[:,:,None]
                        return signal_features(x,True) if kind=='shape' else np.stack([np.ptp(x[:,:,0],axis=1),np.sqrt(np.mean(x[:,:,0]**2,axis=1))],axis=1)
                    yy = np.array([int(r['stick_id']=='WOOD02') for r in pools['train']])
                    fm = fit_centroid(sf(pools['train']),yy)
                    single[f'P{point}:{kind}'] = {name:baseline_metric(np.array([int(r['stick_id']=='WOOD02') for r in rows]),centroid_predict(sf(rows),fm)) for name,rows in pools.items() if name!='train'}
            result = {'source':source,'target':target,'seed':seed,'identity_counts':{k:len(v) for k,v in iparts.items()},'quality_counts':{k:len(v) for k,v in qparts.items()}, 'quality':qresults,'baselines':baselines,'single_point':single,'cnn':{}}
            for mode in ['time','spectrum']:
                cc = {**c,'channels':3,'input_mode':mode,'seed':seed,'epochs':120,'patience':20}
                xt,yt = assembled['train']; xv,yv = assembled['validation']
                tx = inputs(xt,cc); vx = inputs(xv,cc)
                net = CNN(cc)
                history = net.train(np.concatenate([tx,tx[:,:,::-1]]),np.tile(yt,2),np.concatenate([vx,vx[:,:,::-1]]),np.tile(yv,2),cc)
                md = dest/mode; md.mkdir()
                dump(md/'config.json',cc); dump(md/'history.json',history)
                np.savez_compressed(md/'weights.npz',**net.p)  # Frozen before target evaluation.
                evaluation = {}
                for name,(xx,yy) in assembled.items():
                    if name=='train': continue
                    xx = inputs(xx,cc); pp = probabilities(net,xx)
                    rr = probabilities(net,xx[:,:,::-1])
                    assert np.array_equal(pp,rr), '123/321 must give identical fused output'
                    evaluation[name] = scores(yy,pp)
                    predrows = [{'sample_ids':ids,'actual':int(y),'prediction':int(p.argmax()),'probabilities':p.tolist()} for ids,y,p in zip(pairings[name],yy,pp)]
                    dump(md/(name+'_predictions.json'),predrows)
                    if name=='unseen_surface_test':
                        pipeline = {}
                        for profile,gate in gates.items():
                            accepted = np.array([all(gate[sid] for sid in ids) for ids in pairings[name]])
                            pipeline[profile] = {'accepted':int(accepted.sum()),'total':len(accepted),'coverage':float(accepted.mean()),
                                'accepted_accuracy':float((pp.argmax(1)[accepted]==yy[accepted]).mean()) if accepted.any() else None,
                                'correct_and_accepted_rate':float(((pp.argmax(1)==yy)&accepted).mean())}
                        evaluation['pipeline'] = pipeline
                evaluation['epochs_run'] = len(history)
                evaluation['best_epoch'] = min(history,key=lambda r:r['validation_loss'])['epoch']
                evaluation['parameters'] = sum(p.size for p in net.p.values())
                dump(md/'evaluation.json',evaluation); result['cnn'][mode] = evaluation
                print(f"DONE {tag} {mode}: source={evaluation['same_surface_test']['balanced_accuracy']:.4f} target={evaluation['unseen_surface_test']['balanced_accuracy']:.4f} disjoint={evaluation['unseen_surface_disjoint']['balanced_accuracy']:.4f} epochs={len(history)} elapsed={time.monotonic()-start:.1f}s",flush=True)
            all_results.append(result); dump(dest/'summary.json',result); dump(out/'results_partial.json',all_results)
    dump(out/'summary.json',{'protocol':protocol,'experiments':all_results,'elapsed_seconds':time.monotonic()-start})
    print('COMPLETE '+str(out),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    run(p.parse_args().out)


if __name__=='__main__': main()
