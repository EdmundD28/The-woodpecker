"""Normal-only, position-aware single-hit novelty detector; NumPy only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .core import config, dump, preprocess
from .reporting import plotter

EXCLUDED = {
    'WOOD02-E01-R00-P01-H009', 'WOOD02-E01-R00-P01-H027',
    'WOOD02-E01-R00-P01-H038', 'WOOD02-E01-R00-P02-H036',
    'WOOD02-E01-R00-P02-H037',
}
FORMAT = 'WOOD_NORMAL_QUALITY_V1'
FEATURE_NAMES = ([f'post_energy_block_{i}' for i in range(8)] +
                 [f'spectral_band_{i}' for i in range(12)] +
                 ['energy_t10', 'energy_t50', 'energy_t90', 'pre_energy_fraction',
                  'log_crest_factor', 'log_peak_to_peak', 'log_rms'])


def position_group(point):
    if point not in (1, 2, 3):
        raise ValueError('impact point must be 1, 2 or 3')
    return 'middle' if point == 2 else 'endpoint'


def features(record, c):
    x = preprocess(record, c)[:, 0].astype(np.float64)
    post = x[c['trigger_index']:]
    energy = post**2
    total = max(float(energy.sum()), 1e-12)
    temporal = np.array([b.sum()/total for b in np.array_split(energy, 8)])
    power = np.abs(np.fft.rfft(post*np.hanning(len(post))))**2
    power[0] = 0
    bands = np.array([b.sum() for b in np.array_split(power[1:], 12)])
    bands /= max(float(bands.sum()), 1e-12)
    cumulative = np.cumsum(energy)/total
    times = np.searchsorted(cumulative, [.1, .5, .9])/len(post)
    rms = max(float(np.sqrt(np.mean(post**2))), 1e-9)
    peak = max(float(np.abs(post).max()), 1e-9)
    pre_fraction = float(np.sum(x[:c['trigger_index']]**2))/max(float(np.sum(x*x)), 1e-12)
    return np.concatenate([np.log(temporal+1e-6), np.log(bands+1e-6), times,
                           [pre_fraction, np.log(peak/rms), np.log(max(float(np.ptp(x)),1e-9)), np.log(rms)]])


def fit(records, c, k=5):
    """Store only training normals; no wood identity is used during inference."""
    model = {'format_version': FORMAT, 'config': c, 'k': k, 'feature_names': FEATURE_NAMES,
             'groups': {}, 'source': 'normal-only training; two known woods; endpoints pooled'}
    for group in ['endpoint', 'middle']:
        x = np.stack([features(r,c) for r in records if position_group(r['impact_point_id']) == group])
        if len(x) <= k:
            raise ValueError('insufficient normal training records for '+group)
        median = np.median(x, axis=0)
        # Fit location and scale on training normals only; cap sensitivity to near-constant features.
        scale = np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0), .1)
        model['groups'][group] = {'median': median.tolist(), 'scale': scale.tolist(),
                                  'reference': ((x-median)/scale).tolist()}
    return model


def novelty_score(record, model):
    group = model['groups'][position_group(record['impact_point_id'])]
    x = (features(record, model['config'])[:len(group['median'])]-np.asarray(group['median']))/np.asarray(group['scale'])
    distances = np.sqrt(np.mean((np.asarray(group['reference'])-x)**2, axis=1))
    k = model['k']
    return float(np.partition(distances,k-1)[:k].mean())


def calibrate(model, records, alpha=.05):
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie between 0 and 1')
    for name, group in model['groups'].items():
        values = sorted(novelty_score(r,model) for r in records if position_group(r['impact_point_id']) == name)
        rank = math.ceil((len(values)+1)*(1-alpha))
        if rank > len(values) or not values:
            raise ValueError('not enough independent calibration normals for requested alpha')
        group.update(threshold=values[rank-1], calibration_count=len(values), threshold_rank=rank)
    model['target_false_reject_rate'] = alpha
    return model


def predict(record, model):
    if model.get('format_version') != FORMAT:
        raise ValueError('unsupported quality model')
    try:
        preprocess(record,model['config'])
    except (ValueError, TypeError, KeyError) as e:
        return {'sample_id':record.get('sample_id'), 'retest':True,
                'reason':'INVALID_CAPTURE', 'detail':str(e), 'score':None, 'threshold':None}
    name = position_group(record['impact_point_id'])
    threshold = model['groups'][name]['threshold']
    score = novelty_score(record,model)
    return {'sample_id':record['sample_id'], 'retest':score>threshold,
            'reason':'OUTSIDE_NORMAL_RANGE' if score>threshold else 'ACCEPT',
            'score':score,'threshold':threshold,'position_group':name}


def partition(records, seed=42):
    """Stratify normals by specimen and position; abnormal identities never enter pools."""
    rng = np.random.default_rng(seed)
    parts = {name:[] for name in ['train','calibration','normal_test','abnormal_test']}
    strata = {}
    for r in records:
        if r['sample_id'] in EXCLUDED:
            parts['abnormal_test'].append(r)
        else:
            strata.setdefault((r['stick_id'],r['impact_point_id']),[]).append(r)
    for key,pool in sorted(strata.items()):
        pool = sorted(pool,key=lambda r:r['sample_id'])
        order = rng.permutation(len(pool))
        tr,cal = int(len(pool)*.6),int(len(pool)*.2)
        for name,ix in [('train',order[:tr]),('calibration',order[tr:tr+cal]),('normal_test',order[tr+cal:])]:
            parts[name].extend(pool[i] for i in ix)
    return parts


def run(paths,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    c=config(); records=[]; seen=set(); raw_seen=set(); sources=[]
    for p in sorted(map(Path,paths)):
        data=p.read_bytes(); sources.append({'path':str(p.resolve()),'sha256':hashlib.sha256(data).hexdigest()})
        for line in data.decode('utf-8-sig').splitlines():
            if not line.strip(): continue
            r=json.loads(line); preprocess(r,c)
            if r['sample_id'] in seen: raise ValueError('duplicate identity')
            digest=hashlib.sha256(np.asarray(r['raw'],dtype='<u2').tobytes()).hexdigest()
            if digest in raw_seen: raise ValueError('duplicate waveform')
            seen.add(r['sample_id']); raw_seen.add(digest); records.append(r)
    if not EXCLUDED.issubset(seen): raise ValueError('missing named abnormal tests')
    parts=partition(records)
    for name,rows in parts.items():
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    with (out/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_id','stick_id','point','partition','quality_label'])
        for name,rows in parts.items():
            for r in rows: w.writerow([r['sample_id'],r['stick_id'],r['impact_point_id'],name,
                                      'user_identified_abnormal' if name=='abnormal_test' else 'assumed_normal_not_individually_verified'])
    dump(out/'protocol_locked.json',{'sources':sources,'excluded_sample_ids':sorted(EXCLUDED),
         'feature_names':FEATURE_NAMES,'method':'5-nearest-normal mean distance, robust training-only scaling',
         'seed':42,'normal_split':'60/20/remaining within specimen and point',
         'threshold':'separate endpoint/middle split-conformal 5% target, strict >',
         'model_selection':'one predeclared model, no tuning against the five abnormalities or normal test',
         'previous_exposure':'these five waveforms were inspected and discussed before this retrospective experiment'})
    model=calibrate(fit(parts['train'],c),parts['calibration'])
    dump(out/'quality_model.json',model)  # Freeze before opening either test pool.
    results={}
    for name in ['calibration','normal_test','abnormal_test']:
        predictions=[predict(r,model) for r in parts[name]]
        dump(out/(name+'_predictions.json'),predictions)
        rejected=sum(p['retest'] for p in predictions)
        results[name]={'count':len(predictions),'retest':rejected,'accepted':len(predictions)-rejected,
                       'retest_rate':rejected/len(predictions)}
    normal_results=json.loads((out/'normal_test_predictions.json').read_text(encoding='utf-8'))
    by_id={r['sample_id']:r for r in parts['normal_test']}
    subgroups={}
    for p in normal_results:
        r=by_id[p['sample_id']]; key=f"{r['stick_id']}:P{r['impact_point_id']}"
        row=subgroups.setdefault(key,{'count':0,'retest':0}); row['count']+=1; row['retest']+=int(p['retest'])
    summary={'method':model['source'],'counts':{k:len(v) for k,v in parts.items()},'results':results,
             'normal_test_by_specimen_position':subgroups,'thresholds':{k:{field:g[field] for field in ['threshold','calibration_count','threshold_rank']} for k,g in model['groups'].items()},
             'important_limits':['295 remaining records are assumed normal, not independently verified',
                                 'retrospective test: the five abnormal traces were previously inspected',
                                 'no additional unseen abnormal types or new-date data tested',
                                 'target false-reject rate relies on exchangeable calibration and future normals',
                                 'this experiment does not retrain or overwrite the existing wood-identity CNN']}
    dump(out/'summary.json',summary)
    make_report(out,summary,parts,model)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_report(out,s,parts,model):
    normal=s['results']['normal_test']; abnormal=s['results']['abnormal_test']
    text='# 单次敲击质量判定：只学习正常波形\n\n'
    text+=f"**指定异常拦下{abnormal['retest']}/{abnormal['count']}条；假定正常的留出记录误拦{normal['retest']}/{normal['count']}条（{normal['retest_rate']:.1%}）。**\n\n"
    text+='5条指定异常全部独立封存，未进入拟合或阈值校准。其余295条暂按正常处理，不表示已经逐条人工验收。\n\n'
    text+=f"划分：正常训练{s['counts']['train']}条、正常阈值校准{s['counts']['calibration']}条、正常测试{s['counts']['normal_test']}条、指定异常测试5条。原始文件和旧木头识别模型未覆盖。\n\n"
    text+='模型比较新波形与最近的5条正常参考的特征距离；特征包括振动能量时间分布、频带能量、能量累计时间、触发前扰动、波峰因数、幅值及均方根。标准化仅用训练记录拟合。\n\n'
    text+='两个端点共用正常参考，中点单独建模；每个参考池都覆盖两根木头。推理不需要知道木头编号。阈值只用正常校准记录确定，目标误拦率5%；只评估这一套预先固定的方法，没有按5条异常结果反复调参。\n\n'
    text+='| 指定异常 | 是否要求重测 | 异常分数 | 对应阈值 |\n|---|---|---:|---:|\n'
    for r in parts['abnormal_test']:
        p=predict(r,model)
        text+=f"| {r['sample_id']} | {'是' if p['retest'] else '否'} | {p['score']:.3f} | {p['threshold']:.3f} |\n"
    text+='\n正常留出记录分组结果：\n\n| 木头与位置 | 记录数 | 被要求重测 |\n|---|---:|---:|\n'
    for key,row in s['normal_test_by_specimen_position'].items():
        text+=f"| {key} | {row['count']} | {row['retest']} |\n"
    text+='\n被要求重测的假定正常记录：\n\n'
    false_rejects=[predict(r,model) for r in parts['normal_test'] if predict(r,model)['retest']]
    text+='\n'.join('- '+r['sample_id'] for r in false_rejects) if false_rejects else '无。'
    text+='\n\n局限：这是已经看过5条异常后的回顾实验；5条不是5种新异常的泛化证明。其余记录若也有未标出的误采，会影响误拦率解释。新日期、物理翻面、接触变化以及现场重测流程仍需独立实测。5%的目标不保证本次小测试集恰好达到5%。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    plt=plotter(); fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axes,['endpoint','middle']):
        for split,color in [('calibration','gray'),('normal_test','tab:blue'),('abnormal_test','tab:red')]:
            values=[novelty_score(r,model) for r in parts[split] if position_group(r['impact_point_id'])==name]
            ax.scatter(values,[split]*len(values),label=split,color=color,alpha=.8)
        ax.axvline(model['groups'][name]['threshold'],color='black',ls='--',label='retest threshold')
        ax.set(title=name,xlabel='Distance from normal reference'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'quality_results.png',dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    tr=commands.add_parser('train'); tr.add_argument('--data',nargs='+',required=True); tr.add_argument('--out',required=True)
    pr=commands.add_parser('predict'); pr.add_argument('--model',required=True); pr.add_argument('--record',required=True)
    a=p.parse_args()
    if a.command=='train': run(a.data,a.out)
    else:
        model=json.loads(Path(a.model).read_text(encoding='utf-8'))
        record=json.loads(Path(a.record).read_text(encoding='utf-8-sig'))
        print(json.dumps(predict(record,model),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
