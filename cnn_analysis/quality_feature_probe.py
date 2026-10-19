"""One locked source-only quality candidate; anchored identity and duplicate-end stress tests."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .core import config, preprocess
from .cross_surface import SEEDS, split_records, combinations, dump
from .network import CNN, signal_features, centroid_predict
from .quality import EXCLUDED, position_group
from .surface_audit import surface
from .triplet import inputs, probabilities, features as identity_features

SUSPECT='WOOD02-E01-R00-P03-H006'
GOOD=['WOOD02-E01-R00-P03-H033','WOOD02-E01-R00-P03-H038']
BAD=EXCLUDED|{SUSPECT}
CASES=sorted(BAD)+GOOD


def case_features(record,c):
    return signal_features(preprocess(record,c)[None,:,:],True)[0]


def distance(vector,group):
    z=(vector-np.asarray(group['median']))/np.asarray(group['scale'])
    d=np.sqrt(np.mean((np.asarray(group['reference'])-z)**2,axis=1))
    return float(np.partition(d,4)[:5].mean())


def fit_candidate(train,calibration,c):
    # One fixed recipe, inherited 0.1 robust scale floor and 1.5 multiplier.
    # No feature selection, threshold sweep, or target-data optimization.
    model={'feature_count':20,'k':5,'alpha':.05,'multiplier':1.5,'scale_floor':.1,'config':c,'groups':{}}
    for name in ['endpoint','middle']:
        xx=np.stack([case_features(r,c) for r in train if position_group(r['impact_point_id'])==name])
        med=np.median(xx,axis=0);scale=np.maximum(np.quantile(xx,.75,axis=0)-np.quantile(xx,.25,axis=0),.1)
        group={'median':med.tolist(),'scale':scale.tolist(),'reference':((xx-med)/scale).tolist()}
        dd=sorted(distance(case_features(r,c),group) for r in calibration if position_group(r['impact_point_id'])==name)
        rank=math.ceil((len(dd)+1)*.95);assert rank<=len(dd)
        group.update(threshold=dd[rank-1]*1.5,base_threshold=dd[rank-1],rank=rank,calibration_count=len(dd))
        model['groups'][name]=group
    model['training_ids']=[r['sample_id'] for r in train]
    model['calibration_ids']=[r['sample_id'] for r in calibration]
    assert not((set(model['training_ids'])|set(model['calibration_ids']))&set(CASES))
    return model


def check(record,model):
    group=model['groups'][position_group(record['impact_point_id'])]
    score=distance(case_features(record,model['config']),group)
    return {'sample_id':record['sample_id'],'score':score,'threshold':group['threshold'],'retest':score>group['threshold']}


def tensors(triples,records,cache,stress=False):
    for ids in triples:
        rr=[records[sid] for sid in ids]
        assert [r['impact_point_id'] for r in rr]==([3,2,3] if stress else [1,2,3])
        assert len({r['stick_id'] for r in rr})==1
        assert len({surface(r['stick_id'],r['impact_point_id'],r['strike_id']) for r in rr})==1
    return np.stack([np.stack([cache[sid] for sid in ids],axis=1) for ids in triples])


def metrics(predicted,y,accepted):
    correct=predicted==y
    return {'count':len(y),'correct':int(correct.sum()),'accuracy':float(correct.mean()),
            'accepted':int(accepted.sum()),'coverage':float(accepted.mean()),
            'accepted_accuracy':float(correct[accepted].mean()) if accepted.any() else None,
            'accepted_and_correct':int((accepted&correct).sum()),'accepted_and_correct_rate':float((accepted&correct).mean())}


def run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    c=config();records={}
    for src in previous['protocol']['sources']:
        payload=Path(src['path']).read_bytes();assert hashlib.sha256(payload).hexdigest()==src['sha256']
        records.update({r['sample_id']:r for r in map(json.loads,payload.decode('utf-8-sig').splitlines())})
    def support(r):return surface(r['stick_id'],r['impact_point_id'],r['strike_id'])
    protocol={'source_run':str(source),'sources':previous['protocol']['sources'],'seeds':SEEDS,
        'normal_test_cases':GOOD,'suspected_abnormal_case':SUSPECT,'original_abnormal_cases':sorted(EXCLUDED),
        'all_eight_reserved_from_candidate_fit_and_calibration':True,
        'candidate':'single fixed 20-feature normal-neighbor detector: 16 relative spectral bands, four relative quarter energies; no logarithm',
        'position_groups':'middle separately; both endpoints pooled; wood identity not used for quality inference',
        'reference':'source-only fit 40%, calibration ceil40%, remaining same-surface test; five-neighbor mean distance; training median/IQR with fixed .1 floor',
        'threshold':'source-only split-conformal 5% rank; fixed multiplier 1.5 inherited from current venue profile; no target sweep',
        'acceptance':['033 and 038 accepted on all three table-only seeds','all six abnormal/suspected cases rejected on all applicable target-surface seeds',
                      'other target normal false rejection count no worse than current venue gate per seed'],
        'on_failure':'keep current quality implementation and firmware unchanged; do not try another candidate in this run',
        'identity':'frozen source-only time CNN, spectrum CNN, shape centroid from earlier experiment; no fitting or target selection',
        'normal_identity_test':'1000 unique combinations per specimen drawn from target normals, excluding six bad/suspected captures',
        'anchored_identity_test':'each of eight named cases paired with every combination of clean other-position hits from the same wood and surface; keep true point IDs',
        'extra_user_requested_stress':'P03-H033 / normal P02 / P03-H038, actual point IDs 3/2/3; separate from standard 1/2/3 results',
        'limits':['correct identification does not prove physical capture validity','source-only thresholds but feature idea follows inspected examples; retrospective development evidence',
                  'frozen identity models previously treated 006/033/038 as normal when they belonged to their training surface; each named target case is absent from its applicable source-surface training',
                  'combinations share raw hits; no new independent recordings','006 is suspected rather than confirmed abnormal']}
    dump(out/'protocol_locked.json',protocol)
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    all_results=[];predictions=[]
    # Freeze every candidate source-only model before evaluating any target case.
    prepared=[]
    for old in previous['experiments']:
        src=old['source'];target=old['target'];seed=old['seed']
        fit_pool=[r for sid,r in records.items() if support(r)==src and sid not in set(CASES)]
        parts=split_records(fit_pool,seed,quality=True)
        model=fit_candidate(parts['train'],parts['calibration'],c)
        assert all(support(records[sid])==src for sid in model['training_ids']+model['calibration_ids'])
        tag=f'{src}_to_{target}_seed{seed}';dest=out/tag;dest.mkdir()
        dump(dest/'candidate_model.json',model);dump(dest/'source_test_ids.json',[r['sample_id'] for r in parts['test']])
        prepared.append((old,parts,model,dest))
    for old,parts,model,dest in prepared:
        src,target,seed=old['source'],old['target'],old['seed'];tag=dest.name;frozen=source/tag
        target_rows=[r for r in records.values() if support(r)==target]
        candidate={r['sample_id']:check(r,model) for r in target_rows}
        current={}
        for partition in ['unseen_surface_normal','unseen_surface_abnormal']:
            current.update({r['sample_id']:r for r in json.loads((frozen/('venue25_'+partition+'.json')).read_text(encoding='utf-8'))})
        normal=[r for r in target_rows if r['sample_id'] not in BAD]
        target_cases=[sid for sid in CASES if support(records[sid])==target]
        quality={'cases':{sid:{'candidate':candidate[sid],'current_venue':current[sid]} for sid in target_cases},
                 'target_normal_count':len(normal),
                 'target_normal_rejected_candidate':[r['sample_id'] for r in normal if candidate[r['sample_id']]['retest']],
                 'target_normal_rejected_current':[r['sample_id'] for r in normal if current[r['sample_id']]['retest']],
                 'same_surface_normal_test_count':len(parts['test']),
                 'same_surface_normal_test_rejected':[r['sample_id'] for r in parts['test'] if check(r,model)['retest']],
                 'current_venue_leaked_abnormal_ids':[sid for sid in target_cases if sid in BAD and not current[sid]['retest']],
                 'candidate_leaked_abnormal_ids':[sid for sid in target_cases if sid in BAD and not candidate[sid]['retest']]}
        dump(dest/'quality_results.json',quality)
        sets={'normal_random':combinations(normal,1000,191019)}
        clean=[r for r in target_rows if r['sample_id'] not in set(CASES)]
        for sid in target_cases:
            rr=records[sid];pools=[]
            for point in [1,2,3]:
                pools.append([sid] if point==rr['impact_point_id'] else [r['sample_id'] for r in clean if r['stick_id']==rr['stick_id'] and r['impact_point_id']==point])
            sets[sid]=list(itertools.product(*pools))
        if target=='carpet':sets['stress_033_P02_038']=[(GOOD[0],r['sample_id'],GOOD[1]) for r in clean if r['stick_id']=='WOOD02' and r['impact_point_id']==2]
        dump(dest/'triplet_sample_ids.json',sets)
        nc=json.loads((frozen/'time/config.json').read_text(encoding='utf-8'))
        nets={}
        for mode in ['time','spectrum']:
            cc=json.loads((frozen/mode/'config.json').read_text(encoding='utf-8'))
            with np.load(frozen/mode/'weights.npz') as archive:nets[mode]=(CNN(cc,dict(archive)),cc)
        centroid=json.loads((frozen/'shape_without_amplitude_centroid.json').read_text(encoding='utf-8'))
        identity={}
        for name,triples in sets.items():
            xx=tensors(triples,records,cache,stress=name.startswith('stress'))
            yy=np.array([int(records[ids[0]]['stick_id']=='WOOD02') for ids in triples])
            pp={'shape_centroid':centroid_predict(identity_features(xx,'shape_without_amplitude'),centroid)}
            assert np.array_equal(pp['shape_centroid'],centroid_predict(identity_features(xx[:,:,::-1],'shape_without_amplitude'),centroid))
            for mode,(net,cc) in nets.items():
                xxin=inputs(xx,cc);probs=probabilities(net,xxin)
                np.testing.assert_array_equal(probs,probabilities(net,xxin[:,:,::-1]))
                pp[mode]=probs.argmax(1)
            identity[name]={}
            for classifier,pred in pp.items():
                ev={}
                for gate,lookup in [('current_venue',current),('candidate',candidate)]:
                    accepted=np.array([all(not lookup[sid]['retest'] for sid in ids) for ids in triples])
                    ev[gate]=metrics(pred,yy,accepted)
                identity[name][classifier]=ev
                predictions.extend({'experiment':tag,'test':name,'classifier':classifier,'sample_ids':ids,'actual':int(y),'prediction':int(p)} for ids,y,p in zip(triples,yy,pred))
        result={'source':src,'target':target,'seed':seed,'quality':quality,'identity':identity}
        dump(dest/'summary.json',result);all_results.append(result)
        print('DONE '+tag,flush=True)
    normal_pass=all(len(r['quality']['target_normal_rejected_candidate'])<=len(r['quality']['target_normal_rejected_current']) for r in all_results)
    good_pass=all(not p['candidate']['retest'] for r in all_results for sid,p in r['quality']['cases'].items() if sid in GOOD)
    bad_pass=all(p['candidate']['retest'] for r in all_results for sid,p in r['quality']['cases'].items() if sid in BAD)
    decision={'pass':normal_pass and good_pass and bad_pass,'normal_error_count_no_worse':normal_pass,'both_good_accepted_every_seed':good_pass,
              'all_six_bad_suspected_rejected_every_seed':bad_pass,'firmware_modified':False,'quality_implementation_modified':False,
              'further_target_driven_tuning':False}
    dump(out/'summary.json',{'protocol':protocol,'candidate_acceptance':decision,'experiments':all_results})
    with (out/'identity_predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report(out,all_results,decision)
    print(json.dumps(decision,ensure_ascii=False,indent=2))


def report(out,results,decision):
    names={'carpet':'地毯','table':'桌面','time':'时域CNN','spectrum':'频谱CNN','shape_centroid':'三点形态分类器'}
    text='# 033／038识别、重复端点压力测试与一次固定质量候选\n\n'
    text+='全部分类模型来自既有单场地训练，阈值未按目标场地调整；033／038在桌面训练→地毯测试时完全属于未参与训练的场地。每条锚定记录与同木头、同支撑面的其他两点正常记录穷举配组。组合共享原始记录，不是独立采集次数。\n\n'
    text+='## 各单条记录进入标准三点组后，绕过质量拒绝时的识别\n\n| 锚定记录 | 训练→测试 | 每次配组数 | 识别器 | 三次准确率 |\n|---|---|---:|---|---|\n'
    for sid in CASES:
        rr=[r for r in results if sid in r['identity']]
        for classifier in ['shape_centroid','time','spectrum']:
            vals=[r['identity'][sid][classifier]['current_venue'] for r in rr]
            text+=f"| {sid} | {names[rr[0]['source']]}→{names[rr[0]['target']]} | {vals[0]['count']} | {names[classifier]} | "+'、'.join(f"{v['accuracy']:.2%}" for v in vals)+' |\n'
    text+='\n识别正确说明模型在这些配组中仍能利用该记录，不能单凭答案正确证明采集过程没有失误；另两点也可能补足信息。下表六条指定／疑似异常也全部参与识别，不能只检查033和038。\n\n'
    text+='## 033＋点2＋038额外压力测试\n\n真实位置编号为3／2／3，不符合每点各一条的标准协议，单独报告，不计入标准三点准确率。反向交换033和038输出一致。\n\n| 识别器 | 配组数 | 三次识别准确率（质量拒绝前） | 当前宽松档放行组数（三次） |\n|---|---:|---|---|\n'
    for classifier in ['shape_centroid','time','spectrum']:
        vv=[r['identity']['stress_033_P02_038'][classifier]['current_venue'] for r in results if 'stress_033_P02_038' in r['identity']]
        text+=f"| {names[classifier]} | {vv[0]['count']} | "+'、'.join(f"{v['accuracy']:.2%}" for v in vv)+' | '+'、'.join(str(v['accepted']) for v in vv)+' |\n'
    text+='\n## 一个预先固定的质量候选\n\n'
    text+='只尝试一套候选：以16个相对频带能量＋4个相对分段能量替代原质量特征，仍使用5近邻正常参考；训练中位数／四分位距标准化，尺度下限沿用0.1。端点共用参考、中点单独建模；阈值仅用源场地正常校准记录的5%秩确定，再乘既有固定1.5，不搜索目标场地阈值。8条命名记录全部排除出候选拟合和校准。六个源场地模型全部冻结后才评估目标记录。\n\n'
    text+='这是看过案例后提出的新特征开发实验，即使阈值仅用源场地校准，也不是全新独立盲测。\n\n| 波形 | 当前宽松档：三次是否重测 | 候选：三次是否重测 |\n|---|---|---|\n'
    for sid in CASES:
        rr=[r['quality']['cases'][sid] for r in results if sid in r['quality']['cases']]
        text+=f"| {sid} | "+'、'.join('重测' if r['current_venue']['retest'] else '放行' for r in rr)+' | '+'、'.join('重测' if r['candidate']['retest'] else '放行' for r in rr)+' |\n'
    text+='\n| 方向 | 正常测试记录 | 当前宽松误拦数（三次） | 候选误拦数（三次） |\n|---|---:|---|---|\n'
    for src in ['carpet','table']:
        rr=[r for r in results if r['source']==src]
        text+=f"| {names[src]}→{names[rr[0]['target']]} | {rr[0]['quality']['target_normal_count']} | "+'、'.join(str(len(r['quality']['target_normal_rejected_current'])) for r in rr)+' | '+'、'.join(str(len(r['quality']['target_normal_rejected_candidate'])) for r in rr)+' |\n'
    text+='\n当前宽松档对本次六条指定／疑似异常，在各自陌生场地三次测试中是否有漏拦：'+('有，详见逐次清单。' if any(r['quality']['current_venue_leaked_abnormal_ids'] for r in results) else '没有。因此“当前宽松档放行的已知异常”子集为空；没有虚构此类样本。')+'\n\n'
    text+='## 验收结论\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n\n'
    text+='要求同时满足：033与038三次都放行、原5条加006在适用目标方向三次都拦住、其余正常误拦数逐次不增加。本轮只尝试这一套候选，不根据目标结果继续改特征或阈值。未修改质量正式实现、部署权重、原始标签或固件。006始终标记为疑似异常。\n'
    (out/'report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
