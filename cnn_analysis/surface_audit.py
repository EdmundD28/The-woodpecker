"""Retrospective audit using support-surface labels supplied on 2026-10-19."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .quality import EXCLUDED, predict
from .export_quality import venue_model


def surface(stick, point, strike):
    if stick=='WOOD01':
        carpet = {1: strike>=46, 2: strike>=27, 3: strike<=25}[point]
    elif stick=='WOOD02':
        carpet = strike>=26
    else:
        raise ValueError('unknown specimen')
    return 'carpet' if carpet else 'table'


def classify_surfaces(ids, metadata):
    values={metadata[sid]['surface'] for sid in ids}
    return next(iter(values)) if len(values)==1 else 'mixed'


def run(repo,out):
    repo=Path(repo).resolve(); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    identity=repo/'cnn_analysis/runs/20261017_three_point_wood_id'
    quality=repo/'cnn_analysis/runs/20261018_normal_only_quality_v1'
    sources=json.loads((identity/'data_audit.json').read_text(encoding='utf-8'))['files']
    meta={}; files=[]
    for source in sources:
        path=Path(source['path']); payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=source['sha256']:
            raise ValueError('raw file changed since original identity experiment: '+str(path))
        rows=[json.loads(s) for s in payload.decode('utf-8-sig').splitlines() if s.strip()]
        if len(rows)!=50 or [r['strike_id'] for r in rows]!=list(range(1,51)):
            raise ValueError('surface mapping needs 50 sequential hits per file')
        files.append({'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()})
        for ordinal,r in enumerate(rows,1):
            sid=r['sample_id']
            meta[sid]={'sample_id':sid,'stick_id':r['stick_id'],'point':r['impact_point_id'],
                       'strike':r['strike_id'],'surface':surface(r['stick_id'],r['impact_point_id'],ordinal),
                       'quality_label':'user_identified_abnormal' if sid in EXCLUDED else 'assumed_normal',
                       'label_source':'user message 2026-10-19','record':r}
    assert len(meta)==300
    fields=['sample_id','stick_id','point','strike','surface','quality_label','label_source']
    with (out/'surface_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader()
        w.writerows({k:m[k] for k in fields} for m in meta.values())
    counts=defaultdict(Counter)
    for m in meta.values(): counts[f"{m['stick_id']}:P{m['point']}"][m['surface']]+=1
    models={'strict':json.loads((quality/'quality_model.json').read_text(encoding='utf-8')),
            'venue':venue_model(quality/'quality_model.json')}
    qparts={}
    for part in ['train','calibration','normal_test','abnormal_test']:
        for line in (quality/(part+'.jsonl')).read_text(encoding='utf-8').splitlines():
            qparts[json.loads(line)['sample_id']]=part
    qresults={}; predictions=[]
    for name,model in models.items():
        grouped=defaultdict(lambda:{'count':0,'retest':0})
        for sid,m in meta.items():
            p=predict(m['record'],model)
            threshold=p['threshold']*(1.5 if name=='venue' else 1) if p['threshold'] is not None else None
            retest=p['reason']=='INVALID_CAPTURE' or p['score']>threshold
            key=qparts[sid]+':'+m['surface']
            grouped[key]['count']+=1; grouped[key]['retest']+=int(retest)
            predictions.append({'sample_id':sid,'profile':name,'surface':m['surface'],
                                'split':qparts[sid],'score':p['score'],'threshold':threshold,'retest':retest})
        qresults[name]=dict(grouped)
    dump(out/'quality_by_surface_predictions.json',predictions)
    split_records=list(csv.DictReader((identity/'split_manifest.csv').open(encoding='utf-8-sig')))
    raw_pools=defaultdict(Counter); triplet_pools=defaultdict(list)
    for r in split_records:
        m=meta[r['sample_id']]
        raw_pools[r['split']][m['surface']]+=1
        triplet_pools[(r['split'],r['triplet'])].append(r['sample_id'])
    foundation=defaultdict(Counter)
    for (split,_),ids in triplet_pools.items(): foundation[split][classify_surfaces(ids,meta)]+=1
    train_rows=json.loads((identity/'training_combinations.json').read_text(encoding='utf-8'))
    training_mix=dict(Counter(classify_surfaces(ids,meta) for ids in train_rows))
    cnn={}
    for mode in ['time','spectrum']:
        rows=json.loads((identity/mode/'test_predictions.json').read_text(encoding='utf-8'))
        grouped=defaultdict(lambda:{'count':0,'correct':0})
        for r in rows:
            support=classify_surfaces(r['sample_ids'],meta)
            normal=not any(sid in EXCLUDED for sid in r['sample_ids'])
            for key in [support,support+':'+r['actual']]+(['normal_only:'+support] if normal else []):
                grouped[key]['count']+=1; grouped[key]['correct']+=int(r['correct'])
        cnn[mode]=dict(grouped)
    summary={'label_source':'user message 2026-10-19','sources':files,'surface_counts':dict(counts),
             'total_by_surface':dict(Counter(m['surface'] for m in meta.values())),
             'abnormal_by_surface':dict(Counter(meta[sid]['surface'] for sid in EXCLUDED)),
             'quality_by_surface':qresults,'identity_original_raw_split':dict(raw_pools),
             'identity_original_foundation_groups':dict(foundation),
             'identity_training_combinations':training_mix,'identity_test_by_surface':cnn,
             'models_retrained':False,'firmware_modified':False,
             'limits':['both surfaces already represented in original fitting; this is not unseen-surface validation',
                       'same-surface combination counts are correlated combinations of held-out source hits',
                       'WOOD01:P1 carpet pool has only five raw hits total',
                       'remaining records assumed normal rather than individually verified']}
    dump(out/'summary.json',summary)
    write_report(out,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def write_report(out,s):
    names={'table':'桌面','carpet':'地毯','mixed':'混合支撑'}
    text='# 桌面／地毯标签补充与既有模型审计\n\n'
    text+='标签来自2026年10月19日用户说明，按各文件50条的采集顺序映射；已检查编号依次为H001～H050。原始文件、模型权重和固件均未修改。\n\n'
    text+='## 原始数据\n\n| 木头与位置 | 桌面 | 地毯 |\n|---|---:|---:|\n'
    for key,c in s['surface_counts'].items(): text+=f"| {key} | {c.get('table',0)} | {c.get('carpet',0)} |\n"
    text+=f"\n合计桌面{s['total_by_surface']['table']}条、地毯{s['total_by_surface']['carpet']}条。5条指定异常中，桌面{s['abnormal_by_surface']['table']}条、地毯{s['abnormal_by_surface']['carpet']}条。\n\n"
    text+='## 单次质量检查：按真实支撑面分组\n\n| 档位 | 留出桌面正常：要求重测/总数 | 留出地毯正常：要求重测/总数 | 指定异常拦截 |\n|---|---:|---:|---:|\n'
    for profile,title in [('strict','严格'),('venue','默认容错')]:
        r=s['quality_by_surface'][profile]
        table=r['normal_test:table']; carpet=r['normal_test:carpet']
        bad=[v for k,v in r.items() if k.startswith('abnormal_test:')]
        text+=f"| {title} | {table['retest']}/{table['count']} | {carpet['retest']}/{carpet['count']} | {sum(v['retest'] for v in bad)}/{sum(v['count'] for v in bad)} |\n"
    text+='\n训练参考已经混入两种支撑面的正常波形，所以这说明已有两种条件下的留出记录表现，不证明模型能迁移到没有见过的场地或支撑方式。\n\n'
    text+='## 三点配组审计\n\n'
    text+='此前按相同H编号配基础组，不知道中途更换支撑面。木头1三个点的更换位置不一致，因此同H也可能混用桌面与地毯；训练和穷举测试也会产生这种组合。\n\n'
    text+='原训练配组（未计反向增强）：'+str({names[k]:v for k,v in s['identity_training_combinations'].items()})+'。\n\n'
    text+='固定现有权重，仅保留同一种支撑面的原留出测试组合，结果如下：\n\n| 支撑面 | 时域：正确/组合数 | 频谱：正确/组合数 |\n|---|---:|---:|\n'
    for support in ['table','carpet','mixed']:
        a=s['identity_test_by_surface']['time'].get(support,{'count':0,'correct':0})
        b=s['identity_test_by_surface']['spectrum'].get(support,{'count':0,'correct':0})
        text+=f"| {names[support]} | {a['correct']}/{a['count']} | {b['correct']}/{b['count']} |\n"
    text+='\n分木头查看同支撑面组合，避免总体数字掩盖类别差异：\n\n| 木头／支撑面 | 时域：正确/组合数 | 频谱：正确/组合数 |\n|---|---:|---:|\n'
    for stick in ['WOOD01','WOOD02']:
        for support in ['table','carpet']:
            a=s['identity_test_by_surface']['time'].get(support+':'+stick,{'count':0,'correct':0})
            b=s['identity_test_by_surface']['spectrum'].get(support+':'+stick,{'count':0,'correct':0})
            text+=f"| {stick}／{names[support]} | {a['correct']}/{a['count']} | {b['correct']}/{b['count']} |\n"
    text+='\n以上保留原5条异常，如有异常落在测试池也仍计入，不删除错误。可在summary.json查看剔除指定异常后的分组诊断，但不将筛选后的数字替代原成绩。\n\n'
    text+='## 解释与后续边界\n\n'
    text+='桌面与地毯改变机械支撑，既可能改变幅值，也可能改变频谱、共振、衰减和振动传播。不能用简单幅值缩放代替真实支撑变化。\n\n'
    text+='质量模型应允许两种合法条件，不能把“地毯”本身当成误采；现有两种条件已经在训练参考中出现。识别训练应先按木头、支撑面、位置隔离原始记录，再只在同一支撑面内配三点；123/321增强保持。若要评价未见支撑面的迁移能力，应另做桌面训练／地毯测试及反向实验。\n\n'
    text+='木头1点1只有5条地毯记录，数量过少；即使配出很多组合也不增加独立证据。需要补足这一条件，并留出完整的新采集轮次作为最终验收。现有随机划分的成绩不能改称换场地泛化准确率。\n'
    (out/'report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); args=p.parse_args()
    run(Path(__file__).resolve().parents[1],args.out)
