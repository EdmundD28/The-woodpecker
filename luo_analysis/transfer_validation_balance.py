"""Source-only auxiliary control: class-balanced early-stopping validation loss."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .core import preprocess
from .cross_surface import assemble,dump
from .network import CNN
from .transfer_remedies import setup
from .triplet import inputs,probabilities,scores


def run(parent):
    parent=Path(parent);rule=json.loads((parent/'balanced_validation_protocol_locked.json').read_text(encoding='utf-8'))
    assert rule['target_tuning'] is False
    base=Path(json.loads((parent/'protocol_locked.json').read_text(encoding='utf-8'))['base_run'])
    out=parent/'balanced_validation';out.mkdir(exist_ok=False)
    old,records,c=setup(base,out);cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()};frozen=[]
    for seed in rule['seeds']:
        tag=f'carpet_to_table_seed{seed}';src=base/tag;dest=out/tag;dest.mkdir()
        cc=json.loads((src/'spectrum/config.json').read_text(encoding='utf-8'))
        pairs=json.loads((src/'triplet_sample_ids.json').read_text(encoding='utf-8'));dump(dest/'triplet_sample_ids.json',pairs)
        xt,yt=assemble(pairs['train'],records,cache);xv,yv=assemble(pairs['validation'],records,cache)
        tx=inputs(xt,cc);vx=inputs(xv,cc);counts=np.bincount(yv,minlength=2)
        target=math.lcm(*map(int,counts));ids=np.concatenate([np.tile(np.where(yv==k)[0],target//int(counts[k])) for k in [0,1]])
        bx=vx[ids];by=yv[ids];assert np.bincount(by).tolist()==[target,target]
        print('TRAIN balanced source validation '+tag,flush=True)
        net=CNN(cc);history=net.train(np.concatenate([tx,tx[:,:,::-1]]),np.tile(yt,2),np.concatenate([bx,bx[:,:,::-1]]),np.tile(by,2),cc)
        original_history=json.loads((src/'spectrum/history.json').read_text(encoding='utf-8'))
        for row,original_row in zip(history,original_history):
            assert row['train_loss']==original_row['train_loss'] and row['train_accuracy']==original_row['train_accuracy']
        np.savez_compressed(dest/'weights.npz',**net.p);dump(dest/'config.json',cc);dump(dest/'history.json',history)
        best=min(history,key=lambda r:r['validation_loss'])
        original=json.loads((src/'spectrum/evaluation.json').read_text(encoding='utf-8'))
        info={'seed':seed,'source_validation_original_counts':counts.tolist(),'balanced_counts':np.bincount(by).tolist(),
              'best_epoch':best['epoch'],'original_best_epoch':original['best_epoch'],
              'training_trajectory_equal_epochs':min(len(history),len(original_history)),
              'weights_sha256':hashlib.sha256((dest/'weights.npz').read_bytes()).hexdigest()}
        dump(dest/'frozen.json',info);frozen.append(info)
    dump(out/'all_models_frozen.json',{'models':frozen})
    results=[];probes_checked=0
    for info in frozen:
        seed=info['seed'];dest=out/f'carpet_to_table_seed{seed}';cc=json.loads((dest/'config.json').read_text(encoding='utf-8'))
        pairs=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        assert hashlib.sha256((dest/'weights.npz').read_bytes()).hexdigest()==info['weights_sha256']
        with np.load(dest/'weights.npz') as w:net=CNN(cc,dict(w))
        evaluation={}
        for test in ['validation','same_surface_test','unseen_surface_test','unseen_surface_disjoint']:
            xx,yy=assemble(pairs[test],records,cache);xx=inputs(xx,cc);pp=probabilities(net,xx)
            np.testing.assert_array_equal(pp,probabilities(net,xx[:,:,::-1]))
            np.savez_compressed(dest/(test+'_predictions.npz'),probabilities=pp,labels=yy)
            evaluation[test]=scores(yy,pp)
            idx=np.linspace(0,len(yy)-1,min(16,len(yy)),dtype=int)
            np.testing.assert_allclose(probabilities(net,xx[idx]),pp[idx],atol=2e-6,rtol=2e-6);probes_checked+=len(idx)
        baseline=json.loads((base/f'carpet_to_table_seed{seed}'/'spectrum/evaluation.json').read_text(encoding='utf-8'))
        results.append({**info,'evaluation':evaluation,'original_target':baseline['unseen_surface_test'],
                        'delta_percentage_points':100*(evaluation['unseen_surface_test']['balanced_accuracy']-baseline['unseen_surface_test']['balanced_accuracy'])})
    dump(out/'summary.json',{'experiments':results,'mean_delta_percentage_points':float(np.mean([r['delta_percentage_points'] for r in results]))})
    text='# 源验证类别等权：独立因素对照\n\n只改选轮次的验证损失，原始训练池、梯度更新和超参数不变。将原20:80的验证组合重复到80:80，相当于两类验证损失等权。桌面源验证已经平衡，不适用此改动。\n\n| 种子 | 原选轮次 | 等权选轮次 | 原目标平衡准确率 | 等权后 | 变化（百分点） |\n|---|---:|---:|---:|---:|---:|\n'
    for r in results:
        text+=f"| {r['seed']} | {r['original_best_epoch']} | {r['best_epoch']} | {r['original_target']['balanced_accuracy']*100:.2f}% | {r['evaluation']['unseen_surface_test']['balanced_accuracy']*100:.2f}% | {r['delta_percentage_points']:+.2f} |\n"
    text+='\n这是同一批既有目标数据的辅助开发对照，不能视为新采集验证。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    dump(out/'verification.json',{'passed':True,'models_reloaded':len(results),'probabilities_recomputed':probes_checked,'training_rule_unchanged':True})
    print(text,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
