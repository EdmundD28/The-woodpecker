"""Predeclared new-source-split check; reuses targets and is NOT independent confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import preprocess
from .cross_surface import split_records,combinations,assemble,manifest,dump
from .network import fit_centroid,centroid_predict
from .quality import EXCLUDED
from .surface_audit import surface
from .transfer_remedies import setup,fit_preparation,transform,RemedyCNN,band_features,METHODS,CLASSICAL,LABELS
from .triplet import probabilities,scores


def run(parent):
    parent=Path(parent);summary=json.loads((parent/'summary.json').read_text(encoding='utf-8'))
    rule=json.loads((parent/'additional_split_protocol_locked.json').read_text(encoding='utf-8'))
    assert rule['written_before_all_models_frozen']
    rows=summary['experiments'];all_methods=METHODS+CLASSICAL
    directional={method:{s:float(np.mean([x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rows if x['source']==s and x['method']==method])) for s in ['carpet','table']} for method in all_methods}
    candidate=max(all_methods[1:],key=lambda m:min(directional[m].values()))
    eligible=min(directional[candidate].values())>=.95 and min(directional[candidate].values())>=min(directional['raw'].values())+.05
    selected={'candidate':candidate,'eligible':eligible,'directional_development_means':directional,
              'selection_uses_development_target':True,'independent_confirmation':False}
    dump(parent/'additional_split_candidate_locked.json',selected)
    if not eligible:
        print('No candidate meets the predeclared stability-check threshold.',flush=True)
        return
    methods_to_check=['raw',candidate]
    extra_path=parent/'additional_cnn_development_protocol.json'
    if extra_path.exists():
        extra=json.loads(extra_path.read_text(encoding='utf-8'))
        cnn_candidate=max(METHODS[1:],key=lambda m:min(directional[m].values()))
        assert cnn_candidate==extra['candidate'] and extra['selected_after_development_results']
        if cnn_candidate not in methods_to_check: methods_to_check.append(cnn_candidate)
    out=parent/'additional_source_splits';out.mkdir(exist_ok=False)
    base=Path(summary['protocol']['base_run']);old,records,c=setup(base,out)
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()};frozen=[]
    for source,target in [('carpet','table'),('table','carpet')]:
        normals=[r for r in records.values() if r['sample_id'] not in EXCLUDED and surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==source]
        prior=json.loads((base/f'{source}_to_{target}_seed42'/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        for seed in rule['new_source_split_seeds']:
            tag=f'{source}_to_{target}_seed{seed}';dest=out/tag;dest.mkdir()
            cc={**c,'seed':seed};parts=split_records(normals,seed);manifest(parts,dest/'identity_split.csv')
            sets=[{r['sample_id'] for r in rr} for rr in parts.values()]
            assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            prep=fit_preparation(records,sets[0],cc);dump(dest/'preparation.json',prep)
            pairs={'train':combinations(parts['train'],120,seed),'validation':combinations(parts['validation'],80,seed+1),
                   'same_surface_test':combinations(parts['test'],1000,191019),'same_surface_disjoint':combinations(parts['test'],0,191019,True),
                   'unseen_surface_test':prior['unseen_surface_test'],'unseen_surface_disjoint':prior['unseen_surface_disjoint']}
            dump(dest/'triplet_sample_ids.json',pairs)
            xt,yt=assemble(pairs['train'],records,cache);xv,yv=assemble(pairs['validation'],records,cache)
            for method in methods_to_check:
                md=dest/method;md.mkdir();dump(md/'config.json',cc)
                print('TRAIN NEW SPLIT '+tag+' '+method,flush=True)
                if method in METHODS:
                    net=RemedyCNN(cc,method);tx=transform(xt,cc,prep,method);vx=transform(xv,cc,prep,method)
                    history=net.train(np.concatenate([tx,tx[:,:,::-1]]),np.tile(yt,2),np.concatenate([vx,vx[:,:,::-1]]),np.tile(yv,2),cc)
                    np.savez_compressed(md/'weights.npz',**net.p);dump(md/'history.json',history)
                    path=md/'weights.npz';kind='cnn'
                else:
                    bands=int(method.split('_')[0][4:]);fm=fit_centroid(band_features(xt,cc,bands),yt)
                    dump(md/'centroid.json',fm);path=md/'centroid.json';kind='centroid'
                info={'source':source,'target':target,'seed':seed,'method':method,'kind':kind,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                dump(md/'frozen.json',info);frozen.append(info)
    dump(out/'all_models_frozen.json',{'models':frozen,'candidate':candidate,'methods':methods_to_check,'no_independent_target':True})
    results=[];probes_checked=0
    for info in frozen:
        tag=f"{info['source']}_to_{info['target']}_seed{info['seed']}";dest=out/tag;method=info['method'];md=dest/method
        cc=json.loads((md/'config.json').read_text(encoding='utf-8'));prep=json.loads((dest/'preparation.json').read_text(encoding='utf-8'))
        pairs=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        if info['kind']=='cnn':
            path=md/'weights.npz'
            with np.load(path) as w:net=RemedyCNN(cc,method,dict(w))
            def predict(x):return probabilities(net,transform(x,cc,prep,method))
        else:
            path=md/'centroid.json';fm=json.loads(path.read_text(encoding='utf-8'));bands=int(method.split('_')[0][4:])
            def predict(x):return np.eye(2,dtype=np.float32)[centroid_predict(band_features(x,cc,bands),fm)]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['model_sha256']
        evaluation={}
        for test in ['validation','same_surface_test','same_surface_disjoint','unseen_surface_test','unseen_surface_disjoint']:
            xx,yy=assemble(pairs[test],records,cache);pp=predict(xx)
            np.testing.assert_allclose(pp,predict(xx[:,:,::-1]),atol=1e-6,rtol=1e-6)
            np.savez_compressed(md/(test+'_predictions.npz'),probabilities=pp,labels=yy)
            evaluation[test]=scores(yy,pp)
            probes=np.linspace(0,len(yy)-1,min(16,len(yy)),dtype=int)
            np.testing.assert_allclose(predict(xx[probes]),pp[probes],atol=2e-6,rtol=2e-6);probes_checked+=len(probes)
        result={**info,'evaluation':evaluation};dump(md/'evaluation.json',result);results.append(result)
        print(f"NEW SPLIT EVAL {tag} {method} target={evaluation['unseen_surface_test']['balanced_accuracy']:.4f}",flush=True)
    dump(out/'summary.json',{'candidate':candidate,'methods':methods_to_check,'experiments':results,'independent_target_confirmation':False})
    text='# 新源划分稳定性复核\n\n候选按事先写下的规则，从第一阶段的开发目标成绩选择；随后冻结方法，在101、303、707三个新源划分重训比较。目标记录没有变化，因此不是独立新目标验证。\n\n'
    text+=f"冻结候选：{LABELS[candidate]}。\n\n| 方向 | 种子 | 方法 | 目标平衡准确率 | 不复用组平衡准确率 |\n|---|---:|---|---:|---:|\n"
    if len(methods_to_check)>2:
        text=text.replace('冻结候选：', '另追加开发阶段筛出的最佳神经网络，仅检查源划分稳定性，不是独立目标确认。追加规则见additional_cnn_development_protocol.json。\n\n冻结主候选：')
    for r in results:
        e=r['evaluation'];text+=f"| {'地毯→桌面' if r['source']=='carpet' else '桌面→地毯'} | {r['seed']} | {LABELS[r['method']]} | {e['unseen_surface_test']['balanced_accuracy']*100:.2f}% | {e['unseen_surface_disjoint']['balanced_accuracy']*100:.2f}% |\n"
    text+='\n## 三个新划分汇总\n\n| 方向 | 方法 | 目标均值（范围） | 不复用组均值 |\n|---|---|---:|---:|\n'
    for source in ['carpet','table']:
        for method in methods_to_check:
            rr=[x for x in results if x['source']==source and x['method']==method]
            vals=[100*x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rr]
            dis=np.mean([100*x['evaluation']['unseen_surface_disjoint']['balanced_accuracy'] for x in rr])
            text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {LABELS[method]} | {np.mean(vals):.2f}%（{min(vals):.2f}～{max(vals):.2f}） | {dis:.2f}% |\n"
    text+='\n仍需新增、交错采集的真实三点数据确认。没有替换固件或部署权重。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    dump(out/'verification.json',{'passed':True,'models_reloaded':len(frozen),'probabilities_recomputed':probes_checked,'endpoint_exchange_consistency':True,'independent_confirmation':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
