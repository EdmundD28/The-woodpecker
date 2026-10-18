"""Post-hoc joint evaluation of frozen source-only classifiers and quality gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import config, preprocess
from .cross_surface import assemble, dump
from .network import centroid_predict
from .triplet import features


def evaluate(y, predicted, accepted):
    correct=predicted==y
    recall=[]
    for label in [0,1]:
        mask=(y==label)&accepted
        recall.append(float(correct[mask].mean()) if mask.any() else None)
    return {'count':len(y),'accepted':int(accepted.sum()),'coverage':float(accepted.mean()),
            'ungated_accuracy':float(correct.mean()),
            'accepted_accuracy':float(correct[accepted].mean()) if accepted.any() else None,
            'accepted_balanced_accuracy':float(np.mean([r for r in recall if r is not None])) if accepted.any() else None,
            'correct_and_accepted':int((correct&accepted).sum()),
            'correct_and_accepted_rate':float((correct&accepted).mean()),
            'wrong_and_accepted':int((~correct&accepted).sum())}


def span(values):
    return f'{np.mean(values):.2%}（{min(values):.2%}～{max(values):.2%}）'


def run(source, out):
    source=Path(source).resolve(); out=Path(out);out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    records={};c=config()
    for src in summary['protocol']['sources']:
        payload=Path(src['path']).read_bytes();assert hashlib.sha256(payload).hexdigest()==src['sha256']
        records.update({r['sample_id']:r for r in map(json.loads,payload.decode('utf-8-sig').splitlines())})
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    rows=[]
    for e in summary['experiments']:
        dest=source/f"{e['source']}_to_{e['target']}_seed{e['seed']}"
        pairings=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        for partition in ['unseen_surface_test','unseen_surface_disjoint']:
            triples=pairings[partition];x,y=assemble(triples,records,cache)
            centroid=json.loads((dest/'shape_without_amplitude_centroid.json').read_text(encoding='utf-8'))
            predictions={'shape_centroid':centroid_predict(features(x,'shape_without_amplitude'),centroid)}
            for mode in ['time','spectrum']:
                saved=json.loads((dest/mode/(partition+'_predictions.json')).read_text(encoding='utf-8'))
                assert [p['sample_ids'] for p in saved]==triples and [p['actual'] for p in saved]==y.tolist()
                predictions[mode]=np.array([p['prediction'] for p in saved])
            for gate in ['hard_only','strict27','venue25']:
                if gate=='hard_only':
                    # All 300 raw records passed existing protocol, clipping, loss-flag and weak-signal checks.
                    accepted=np.ones(len(y),dtype=bool)
                    normal_false=0;abnormal_detected=0
                    abnormal_count=e['quality']['venue25']['unseen_surface_abnormal']['all']['count']
                else:
                    q=json.loads((dest/(gate+'_unseen_surface_normal.json')).read_text(encoding='utf-8'))
                    byid={p['sample_id']:not p['retest'] for p in q}
                    accepted=np.array([all(byid[sid] for sid in triple) for triple in triples])
                    normal_false=sum(p['retest'] for p in q)
                    abnormal=e['quality'][gate]['unseen_surface_abnormal']['all']
                    abnormal_detected=abnormal['retest'];abnormal_count=abnormal['count']
                for classifier,predicted in predictions.items():
                    rows.append(dict(source=e['source'],target=e['target'],seed=e['seed'],partition=partition,
                        classifier=classifier,gate=gate,normal_false_rejections=normal_false,
                        target_normal_hits=summary['protocol']['normal_counts'][e['target']],
                        target_abnormal_detected=abnormal_detected,target_abnormal_count=abnormal_count,
                        **evaluate(y,predicted,accepted)))
    dump(out/'summary.json',{'source_run':str(source),'post_hoc_joint_analysis':True,'fitting_or_tuning':False,
         'method':'join existing frozen per-hit quality decisions with the same held-out target triplets; no new threshold selection',
         'source_protocol_sha256':hashlib.sha256((source/'protocol_locked.json').read_bytes()).hexdigest(),
         'rows':rows,'limits':['repeated combinations are correlated','only two known specimens and existing two surfaces',
                              'same previously inspected five abnormal captures, one table and four carpet',
                              'all remaining 295 hits assumed normal','no firmware changes or actual retries simulated']})
    with (out/'joint_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    names={'carpet':'地毯','table':'桌面','time':'时域CNN','spectrum':'频谱CNN','shape_centroid':'三点形态分类器',
           'hard_only':'仅现有硬故障检查','strict27':'严格质量档','venue25':'当前宽松质量档'}
    text='# 陌生支撑面：质量检查与识别的完整流程\n\n'
    text+='本次只组合此前冻结的分类器与质量判定结果，没有重新训练、选模型或调整阈值；属于既有测试集的事后联合分析，不是新增独立验证。\n\n'
    text+='指标分别回答：正常三点组能否直接放行、放行以后能否认对木头、全部正常组中有多少既放行又认对。拒绝会要求重测，因此“直接放行且判对”描述一次尝试，不等于最终经过重测后的准确率。\n\n'
    for partition,title in [('unseen_surface_test','随机组合'),('unseen_surface_disjoint','不重复使用原始记录的三点组')]:
        text+='## '+title+'\n\n| 训练→测试 | 识别器 | 质量档 | 三点直接放行率 | 放行后准确率 | 全部组直接放行且判对 |\n|---|---|---|---:|---:|---:|\n'
        for direction in ['carpet','table']:
            for classifier in ['shape_centroid','time','spectrum']:
                for gate in ['hard_only','strict27','venue25']:
                    rr=[r for r in rows if r['source']==direction and r['classifier']==classifier and r['gate']==gate and r['partition']==partition]
                    text+=f"| {names[direction]}→{names[rr[0]['target']]} | {names[classifier]} | {names[gate]} | {span([r['coverage'] for r in rr])} | {span([r['accepted_accuracy'] for r in rr if r['accepted_accuracy'] is not None])} | {span([r['correct_and_accepted_rate'] for r in rr])} |\n"
        text+='\n'
    text+='## 质量检查自身的漏拦与误拦\n\n| 方向 | 档位 | 陌生场地正常误拦数（三次） | 陌生场地指定异常拦截（三次） |\n|---|---|---|---|\n'
    for direction in ['carpet','table']:
        for gate in ['hard_only','strict27','venue25']:
            rr=[r for r in rows if r['source']==direction and r['classifier']=='shape_centroid' and r['gate']==gate and r['partition']=='unseen_surface_test']
            normal='、'.join(f"{r['normal_false_rejections']}/{r['target_normal_hits']}" for r in rr)
            bad='、'.join(f"{r['target_abnormal_detected']}/{r['target_abnormal_count']}" for r in rr)
            text+=f"| {names[direction]}→{names[rr[0]['target']]} | {names[gate]} | {normal} | {bad} |\n"
    text+='\n现有5条指定异常都通过了原先的协议、丢样标记、削顶和弱信号检查，因此只保留这些硬检查会放过全部5条。当前宽松档在本批数据中取得较低正常误拦，但不能据此承诺任意新支撑面零误拦或未知异常零漏拦。严格档的高误拦说明正常分布距离也在响应场地变化；质量检查学到的正常范围不天然等同于采集是否有效。\n\n'
    text+='随机测试每根木头1000组，重复使用原始记录；地毯不重复组只有28组，其中木头1仅5组。跨场地还与采集时间顺序混杂，剩余记录只是暂按正常。表中放行后准确率按放行组合实际数量计算，平衡准确率另存CSV。没有覆盖原始数据、模型或固件。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
