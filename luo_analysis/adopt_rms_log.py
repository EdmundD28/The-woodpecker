"""Train the adopted mixed-support RMS/log triplet model with locked seed and splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .cross_surface import assemble,dump
from .core import preprocess
from .network import CNN
from .spectrum_preparation import VERSION
from .transfer_remedies import load_records,fit_preparation
from .triplet import FORMAT,POINTS,inputs,contract,probabilities,scores,export_triplet,predict


def run(base,out):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    old=json.loads((base/'summary.json').read_text(encoding='utf-8'));records=load_records(old)
    c=json.loads((base/'carpet_to_table_seed42/spectrum/config.json').read_text(encoding='utf-8'))
    c.update(channels=1,seed=42)
    protocol={'method':'rms_log','seed':42,'sources':old['protocol']['sources'],
              'selection':'One model, user-adopted architecture; unweighted clean validation loss, as in remedy study. No added balanced-validation combination.',
              'data':'Both known supports trained together; exact source seed42 training/validation/test partitions from each support. Five prior exclusions unchanged.',
              'test_scope':'Previously inspected development recordings; this is NOT an independent new deployment acceptance test.',
              'normalization_fit':'unique training hits only, endpoint parameters pooled',
              'architecture':c['conv_layers'],'epochs':c['epochs'],'learning_rate':c['learning_rate']}
    dump(out/'protocol_locked.json',protocol)
    pairs={};parts={};by_surface={}
    for source,target in [('carpet','table'),('table','carpet')]:
        src=base/f'{source}_to_{target}_seed42'
        with (src/'identity_split.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                sid=row['sample_id'];parts.setdefault(row['partition'],set()).add(sid)
        original=json.loads((src/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        by_surface[source]={}
        for name,key in [('train','train'),('validation','validation'),('test','same_surface_test'),('disjoint_test','same_surface_disjoint')]:
            pairs.setdefault(name,[]).extend(original[key]);by_surface[source][name]=original[key]
    assert not(parts['train']&parts['validation'] or parts['train']&parts['test'] or parts['validation']&parts['test'])
    assert len(set.union(*parts.values()))==295
    dump(out/'split_ids.json',{k:sorted(v) for k,v in parts.items()});dump(out/'triplet_sample_ids.json',pairs);dump(out/'pairings_by_surface.json',by_surface)
    preparation=fit_preparation(records,parts['train'],c);dump(out/'preparation_fit.json',preparation)
    c['spectrum_preparation']={k:preparation[k] for k in ['rms_target','log_reference','log_gain']}
    c['spectrum_preparation']['version']=VERSION
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    xt,yt=assemble(pairs['train'],records,cache);xv,yv=assemble(pairs['validation'],records,cache)
    tx=inputs(xt,c);vx=inputs(xv,c);net=CNN({**c,'channels':3})
    print('TRAIN adopted RMS/log model',flush=True)
    history=net.train(np.concatenate([tx,tx[:,:,::-1]]),np.tile(yt,2),np.concatenate([vx,vx[:,:,::-1]]),np.tile(yv,2),c)
    weights={k:v.tolist() for k,v in net.p.items()}
    digest=hashlib.sha256(json.dumps({'preparation':c['spectrum_preparation'],'weights':weights},sort_keys=True).encode()).hexdigest()[:12]
    model={'format_version':FORMAT,'config':c,'network_channels':3,'point_order':POINTS,'input_contract':contract(c),
           'weights':weights,'model_version':'REAL-RMSLOG-'+digest,'status':'ADOPTED_HOST_VERIFICATION_PENDING',
           'source_audit':old['protocol']['sources'],'adoption_basis':'User selected RMS normalization + log spectrum based on retrospective transfer study.',
           'independent_new_acceptance':False}
    child=out/'spectrum';child.mkdir();dump(child/'model.json',model);dump(child/'history.json',history)
    np.savez_compressed(child/'weights.npz',**net.p)
    export_triplet(model,child)
    dump(out/'model_frozen_before_test.json',{'model_version':model['model_version'],'weights_sha256':hashlib.sha256((child/'weights.npz').read_bytes()).hexdigest()})
    evaluation={}
    for source in ['carpet','table','pooled']:
        ppairs=pairs if source=='pooled' else by_surface[source]
        evaluation[source]={}
        for name in ['validation','test','disjoint_test']:
            xx,yy=assemble(ppairs[name],records,cache);xx=inputs(xx,c);p=probabilities(net,xx)
            np.testing.assert_array_equal(p,probabilities(net,xx[:,:,::-1]))
            evaluation[source][name]=scores(yy,p)
            np.savez_compressed(child/f'{source}_{name}_predictions.npz',probabilities=p,labels=yy)
    vectors=[]
    for source in ['carpet','table']:
        triples=by_surface[source]['disjoint_test']
        for wood in ['WOOD01','WOOD02']:
            selected=[ids for ids in triples if records[ids[0]]['stick_id']==wood][:2]
            for ids in selected:
                xx,_=assemble([ids],records,cache);xx=inputs(xx,c)
                rr=[records[sid] for sid in ids]
                expected=probabilities(net,xx)[0]
                np.testing.assert_allclose(predict(child/'model.json',rr)['scores'],expected,atol=1e-7)
                np.testing.assert_allclose(predict(child/'model.json',rr[::-1])['scores'],expected,atol=1e-7)
                vectors.append({'surface':source,'records':rr,'preprocessed_input':xx[0].tolist(),'expected_scores':expected.tolist()})
    dump(child/'fixed_test_vectors.json',vectors);dump(child/'evaluation.json',evaluation)
    summary={'model_version':model['model_version'],'raw_partition_counts':{k:len(v) for k,v in parts.items()},
             'combination_counts':{k:len(v) for k,v in pairs.items()},'epochs':len(history),
             'best_epoch':min(history,key=lambda r:r['validation_loss'])['epoch'],'parameters':sum(v.size for v in net.p.values()),
             'oracle_vectors':len(vectors),'evaluation':evaluation,'independent_new_acceptance':False}
    dump(out/'summary.json',summary)
    text='# 采用版：均方根归一化＋对数频谱卷积网络\n\n'
    text+=f"模型：{model['model_version']}；参数{summary['parameters']}。步长仍为2/2/2，末层展平保留频率位置。逐击均方根归一化后，使用Hann窗、2400点频谱幅值与拟合的log1p压缩；全部统计只拟合训练池，两个端点共用参数。\n\n"
    text+=f"桌面与地毯共同训练，原始敲击划分为{summary['raw_partition_counts']}；三点组合数为{summary['combination_counts']}。种子42和训练规则在看本轮成绩前固定。未将未经组合试验的等权验证修改叠加到本模型。\n\n"
    text+='| 支撑面 | 验证平衡准确率 | 留出组合平衡准确率 | 不复用组平衡准确率 | 不复用组数量 |\n|---|---:|---:|---:|---:|\n'
    for source,label in [('carpet','地毯'),('table','桌面'),('pooled','合并')]:
        e=evaluation[source];text+=f"| {label} | {e['validation']['balanced_accuracy']:.2%} | {e['test']['balanced_accuracy']:.2%} | {e['disjoint_test']['balanced_accuracy']:.2%} | {e['disjoint_test']['count']} |\n"
    text+='\n这些留出记录已被此前开发研究查看，因此是开发复核，不能替代新增现场验收；共同训练的成绩也不能当作单场地训练的跨场地测试成绩。原始记录和旧时域模型保留。模型预测、导出参考与固件FFT须分别数值核对，真机时延和新场地表现仍需实测。\n'
    (out/'report.md').write_text(text,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='luo_analysis/runs/20261019_cross_surface_v1');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.base,a.out)
