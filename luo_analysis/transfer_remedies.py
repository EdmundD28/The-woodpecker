"""Locked retrospective transfer diagnosis and source-only remedy comparisons."""
import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from .core import preprocess
from .cross_surface import assemble, dump
from .network import CNN, fit_centroid, centroid_predict
from .quality import EXCLUDED
from .surface_audit import surface
from .triplet import inputs, probabilities, scores

METHODS=['raw','rms','spectral_l1','rms_log','gain_aug','rms_shape_aug']
CLASSICAL=['band16_centroid','band64_centroid']
LABELS={'raw':'原频谱网络','rms':'逐击均方根归一化','spectral_l1':'逐击频谱幅值和归一化',
        'rms_log':'均方根归一化＋对数压缩','gain_aug':'独立力度增强',
        'rms_shape_aug':'均方根归一化＋频谱形状增强',
        'band16_centroid':'16频带相对能量＋类中心','band64_centroid':'64频带相对能量＋类中心'}


def load_records(old):
    records={}
    for src in old['protocol']['sources']:
        payload=Path(src['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==src['sha256']
        for line in payload.decode('utf-8-sig').splitlines():
            if line.strip():
                r=json.loads(line); assert r['sample_id'] not in records
                records[r['sample_id']]=r
    assert len(records)==300 and EXCLUDED<=set(records)
    return records


def symmetric_stat(values,points,stat=np.median):
    end=float(stat(values[np.isin(points,[1,3])]))
    middle=float(stat(values[points==2]))
    return np.array([end,middle,end],dtype=np.float64)


def fit_preparation(records,train_ids,c):
    rr=[records[sid] for sid in sorted(train_ids)]
    tt=np.stack([preprocess(r,c)[:,0] for r in rr]).astype(np.float64)
    pp=np.array([r['impact_point_id'] for r in rr])
    rms=np.sqrt(np.mean(tt*tt,axis=1))
    target=symmetric_stat(rms,pp)
    raw_spec=np.abs(np.fft.rfft(tt*np.hanning(c['sample_count']),axis=1))/c['sample_count']
    l1=symmetric_stat(raw_spec.sum(1),pp)
    norm=tt*target[pp-1,None]/np.maximum(rms[:,None],1e-12)
    spec=np.abs(np.fft.rfft(norm*np.hanning(c['sample_count']),axis=1))/c['sample_count']
    reference=symmetric_stat(spec.mean(1),pp)
    compressed=np.log1p(spec/np.maximum(reference[pp-1,None],1e-12))
    # Restore a source-only typical input scale so log compression is not just a learning-rate change.
    ratio=np.sqrt(np.mean(spec*spec,1))/np.maximum(np.sqrt(np.mean(compressed*compressed,1)),1e-12)
    log_gain=symmetric_stat(ratio,pp)
    return {'fit_sample_ids':sorted(train_ids),'rms_target':target.tolist(),'spectral_l1_target':l1.tolist(),
            'log_reference':reference.tolist(),'log_gain':log_gain.tolist(),
            'pooling':'unique source training hits, endpoints pooled; no validation or target fitting'}


def transform(time_inputs,c,prep,method):
    x=time_inputs.astype(np.float64)
    if method in ['rms','rms_log','rms_shape_aug']:
        rms=np.sqrt(np.mean(x*x,axis=1,keepdims=True))
        x=x*np.asarray(prep['rms_target'])[None,None,:]/np.maximum(rms,1e-12)
    spec=inputs(x.astype(np.float32),c)
    if method=='spectral_l1':
        spec=(spec*np.asarray(prep['spectral_l1_target'])[None,None,:]/np.maximum(spec.astype(np.float64).sum(1,keepdims=True),1e-12)).astype(np.float32)
    elif method=='rms_log':
        spec=(np.log1p(spec.astype(np.float64)/np.maximum(np.asarray(prep['log_reference'])[None,None,:],1e-12))*np.asarray(prep['log_gain'])[None,None,:]).astype(np.float32)
    return spec


class RemedyCNN(CNN):
    """Same initialization and optimizer; augmentation changes only each training mini-batch."""
    def __init__(self,c,method,weights=None):
        super().__init__(c,weights)
        self.method=method
        self.aug_rng=np.random.default_rng(c['seed']+191019)

    def augment(self,x):
        if self.method=='gain_aug':
            gain=np.exp(self.aug_rng.uniform(-np.log(2),np.log(2),size=(len(x),1,3)))
            return (x*gain).astype(np.float32)
        if self.method=='rms_shape_aug':
            # A declared nuisance family, not a fitted model of a particular support surface.
            knots=self.aug_rng.normal(0,.35,size=(len(x),6,3))
            axis=np.arange(x.shape[1]); grid=np.linspace(0,x.shape[1]-1,6)
            response=np.empty(x.shape,dtype=np.float32)
            for n in range(len(x)):
                for p in range(3): response[n,:,p]=np.exp(np.interp(axis,grid,knots[n,:,p]))
            shifted=x*response
            before=np.sqrt(np.sum(x.astype(np.float64)**2,axis=1,keepdims=True))
            after=np.sqrt(np.sum(shifted.astype(np.float64)**2,axis=1,keepdims=True))
            return (shifted*before/np.maximum(after,1e-12)).astype(np.float32)
        return x

    def gradients(self,x,y):
        return super().gradients(self.augment(x),y)


def band_features(time_inputs,c,bands):
    spec=inputs(time_inputs,c).astype(np.float64)
    power=spec[:,1:,:]**2  # DC excluded, all remaining bins including Nyquist retained.
    relative=np.stack([a.sum(1) for a in np.array_split(power,bands,axis=1)],axis=1)
    relative/=np.maximum(relative.sum(1,keepdims=True),1e-12)
    return np.concatenate([(relative[:,:,0]+relative[:,:,2])*.5,relative[:,:,1]],axis=1)


def diagnosis(records,c,out):
    """Descriptive diagnostics only; never supplies fitting parameters to a remedy."""
    groups={}
    for sid,r in records.items():
        if sid in EXCLUDED: continue
        x=preprocess(r,c)[:,0].astype(np.float64)
        spec=np.abs(np.fft.rfft(x*np.hanning(len(x))))/len(x)
        power=spec[1:]**2
        bands=np.array([a.sum() for a in np.array_split(power,16)])
        bands/=max(bands.sum(),1e-12)
        centroid=float(np.sum(np.fft.rfftfreq(len(x),1/c['sample_rate_hz'])[1:]*power)/max(power.sum(),1e-12))
        key=(r['stick_id'],r['impact_point_id'],surface(r['stick_id'],r['impact_point_id'],r['strike_id']))
        groups.setdefault(key,[]).append({'id':sid,'rms':float(np.sqrt(np.mean(x*x))),'peak':float(np.max(np.abs(x))),
                                         'centroid':centroid,'bands':bands})
    result=[]
    for (wood,point,support),rr in sorted(groups.items()):
        result.append({'wood':wood,'point':point,'surface':support,'count':len(rr),
                       'rms_median':float(np.median([x['rms'] for x in rr])),
                       'rms_q10_q90':np.quantile([x['rms'] for x in rr],[.1,.9]).tolist(),
                       'centroid_median_hz':float(np.median([x['centroid'] for x in rr])),
                       'mean_relative_power16':np.mean([x['bands'] for x in rr],axis=0).tolist()})
    comparisons=[]
    for wood,point in itertools.product(['WOOD01','WOOD02'],[1,2,3]):
        a=next(r for r in result if (r['wood'],r['point'],r['surface'])==(wood,point,'carpet'))
        b=next(r for r in result if (r['wood'],r['point'],r['surface'])==(wood,point,'table'))
        # Total variation between mean relative band energies, independent of absolute gain.
        shift=float(np.sum(np.abs(np.array(a['mean_relative_power16'])-b['mean_relative_power16']))*.5)
        comparisons.append({'wood':wood,'point':point,'carpet_count':a['count'],'table_count':b['count'],
                            'table_over_carpet_rms_median':b['rms_median']/a['rms_median'],
                            'relative_band_power_total_variation':shift,
                            'table_minus_carpet_centroid_hz':b['centroid_median_hz']-a['centroid_median_hz']})
    contrasts=[]
    for point in [1,2,3]:
        contrast={}
        for support in ['carpet','table']:
            a=next(r for r in result if (r['wood'],r['point'],r['surface'])==('WOOD01',point,support))
            b=next(r for r in result if (r['wood'],r['point'],r['surface'])==('WOOD02',point,support))
            contrast[support]=np.array(b['mean_relative_power16'])-a['mean_relative_power16']
        cosine=float(np.dot(contrast['carpet'],contrast['table'])/max(np.linalg.norm(contrast['carpet'])*np.linalg.norm(contrast['table']),1e-12))
        contrasts.append({'point':point,'wood_separation_tv_carpet':float(np.abs(contrast['carpet']).sum()*.5),
                          'wood_separation_tv_table':float(np.abs(contrast['table']).sum()*.5),
                          'wood_contrast_direction_cosine_between_surfaces':cosine})
    dump(out/'signal_diagnosis.json',{'groups':result,'cross_surface_comparisons':comparisons,'wood_contrasts':contrasts,
         'interpretation':'Descriptive association; surface is confounded with acquisition order, force and contact. Not a causal physical identification.'})


def setup(base,out):
    old=json.loads((base/'summary.json').read_text(encoding='utf-8'))
    records=load_records(old)
    c=json.loads((base/'carpet_to_table_seed42/spectrum/config.json').read_text(encoding='utf-8'))
    return old,records,c


def train(base,out):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    old,records,c=setup(base,out)
    protocol={'base_run':str(base.resolve()),'base_summary_sha256':hashlib.sha256((base/'summary.json').read_bytes()).hexdigest(),
              'methods':METHODS,'classical':CLASSICAL,'seeds':[42,123,2026],
              'numpy':np.__version__,'python':sys.version,'python_executable':sys.executable,
              'rules':['Retrospective development data, already inspected; not a new blind test',
                       'exact prior split manifests, exclusions and triplets; no quality-gate filtering',
                       'prepare statistics from unique source training hits only; no target normalization fitting',
                       'all CNNs share stride222 architecture, initial weights, optimizer, batch order, steps and early-stop budget',
                       'augmentation has a separate RNG and operates inside each existing training batch, no extra steps',
                       'gain augmentation: independent loguniform [0.5,2] per point per batch',
                       'shape augmentation: six fixed-grid log-response knots, normal sd0.35, independent per hit; preserve each spectrum L2 norm',
                       'two classical models use only normalized spectral band power, no time-domain features',
                       'freeze all models and source-validation selections before any new target score',
                       'all methods/seeds/directions reported; target results never tune hyperparameters',
                       'combination counts and three seeds are not independent acquisitions',
                       'no firmware replacement; collect new support-balanced sessions for final confirmation']}
    dump(out/'protocol_locked.json',protocol)
    diagnosis(records,c,out)
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    frozen=[]; start=time.monotonic()
    for exp in old['experiments']:
        tag=f"{exp['source']}_to_{exp['target']}_seed{exp['seed']}";src=base/tag;dest=out/tag;dest.mkdir()
        cc=json.loads((src/'spectrum/config.json').read_text(encoding='utf-8'))
        for fn in ['identity_split.csv','triplet_sample_ids.json']: (dest/fn).write_bytes((src/fn).read_bytes())
        with (src/'identity_split.csv').open(encoding='utf-8-sig',newline='') as f:
            train_ids={r['sample_id'] for r in csv.DictReader(f) if r['partition']=='train'}
        prep=fit_preparation(records,train_ids,cc); dump(dest/'preparation.json',prep)
        pairs=json.loads((src/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        xt,yt=assemble(pairs['train'],records,cache);xv,yv=assemble(pairs['validation'],records,cache)
        val_losses={}; initial=None
        for method in METHODS:
            md=dest/method;md.mkdir();net=RemedyCNN(cc,method)
            if initial is None: initial={k:v.copy() for k,v in net.p.items()}
            else:
                for k in initial: np.testing.assert_array_equal(net.p[k],initial[k])
            tx=transform(xt,cc,prep,method);vx=transform(xv,cc,prep,method)
            np.testing.assert_allclose(transform(xt[:,:,::-1],cc,prep,method),tx[:,:,::-1],atol=1e-7,rtol=1e-6)
            t=time.monotonic();print('TRAIN '+tag+' '+method,flush=True)
            history=net.train(np.concatenate([tx,tx[:,:,::-1]]),np.tile(yt,2),np.concatenate([vx,vx[:,:,::-1]]),np.tile(yv,2),cc)
            if method=='raw':
                with np.load(src/'spectrum/weights.npz') as w:
                    for k in w.files: np.testing.assert_array_equal(net.p[k],w[k])
                assert history==json.loads((src/'spectrum/history.json').read_text(encoding='utf-8'))
            dump(md/'config.json',cc);dump(md/'history.json',history);np.savez_compressed(md/'weights.npz',**net.p)
            best=min(history,key=lambda r:r['validation_loss'])
            info={'source':exp['source'],'target':exp['target'],'seed':exp['seed'],'method':method,'kind':'cnn',
                  'validation':scores(yv,probabilities(net,vx)), 'selection_loss':best['validation_loss'],
                  'best_epoch':best['epoch'],'epochs':len(history),'training_seconds':time.monotonic()-t,
                  'parameters':sum(p.size for p in net.p.values()),
                  'weights_sha256':hashlib.sha256((md/'weights.npz').read_bytes()).hexdigest()}
            dump(md/'frozen.json',info);frozen.append(info);val_losses[method]=info['selection_loss']
            print(f"FROZEN {tag} {method} val={info['selection_loss']:.6f} seconds={info['training_seconds']:.1f}",flush=True)
        for method in CLASSICAL:
            bands=int(method.split('_')[0][4:]);md=dest/method;md.mkdir()
            model=fit_centroid(band_features(xt,cc,bands),yt);dump(md/'centroid.json',model);dump(md/'config.json',cc)
            pred=centroid_predict(band_features(xv,cc,bands),model)
            info={'source':exp['source'],'target':exp['target'],'seed':exp['seed'],'method':method,'kind':'centroid',
                  'validation':scores(yv,np.eye(2)[pred]),'bands':bands,
                  'centroid_sha256':hashlib.sha256((md/'centroid.json').read_bytes()).hexdigest()}
            dump(md/'frozen.json',info);frozen.append(info)
        dump(dest/'selection_locked.json',{'cnn_loss':val_losses,'cnn_selected':min(val_losses,key=val_losses.get),
             'criterion':'CNN minimum clean source validation loss; classical maximum source validation balanced accuracy, tie uses 16 bands',
             'classical_selected':max(CLASSICAL,key=lambda m:next(r for r in frozen if r['source']==exp['source'] and r['seed']==exp['seed'] and r['method']==m)['validation']['balanced_accuracy'])})
    dump(out/'all_models_frozen.json',{'models':frozen,'training_elapsed_seconds':time.monotonic()-start})
    evaluate(base,out)


def evaluate(base,out):
    base=Path(base);out=Path(out);old,records,c=setup(base,out)
    frozen=json.loads((out/'all_models_frozen.json').read_text(encoding='utf-8'))
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()};results=[]
    for info in frozen['models']:
        method=info['method'];tag=f"{info['source']}_to_{info['target']}_seed{info['seed']}";dest=out/tag;md=dest/method
        cc=json.loads((md/'config.json').read_text(encoding='utf-8'));prep=json.loads((dest/'preparation.json').read_text(encoding='utf-8'))
        pairs=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        if info['kind']=='cnn':
            assert hashlib.sha256((md/'weights.npz').read_bytes()).hexdigest()==info['weights_sha256']
            with np.load(md/'weights.npz') as w: net=RemedyCNN(cc,method,dict(w))
            def predict(x): return probabilities(net,transform(x,cc,prep,method))
        else:
            assert hashlib.sha256((md/'centroid.json').read_bytes()).hexdigest()==info['centroid_sha256']
            fm=json.loads((md/'centroid.json').read_text(encoding='utf-8'))
            def predict(x): return np.eye(2,dtype=np.float32)[centroid_predict(band_features(x,cc,info['bands']),fm)]
        evaluation={}
        for test in ['validation','same_surface_test','unseen_surface_test','same_surface_disjoint','unseen_surface_disjoint']:
            xx,yy=assemble(pairs[test],records,cache);pp=predict(xx)
            np.testing.assert_allclose(pp,predict(xx[:,:,::-1]),atol=1e-6,rtol=1e-6)
            evaluation[test]=scores(yy,pp)
            np.savez_compressed(md/(test+'_predictions.npz'),probabilities=pp,labels=yy)
            if test=='unseen_surface_disjoint':
                sensitivity=[]
                for gains in [[.5,.5,.5],[2,2,2],[.5,2,1.5]]:
                    changed=predict((xx*np.array(gains)[None,None,:]).astype(np.float32))
                    sensitivity.append({'gains':gains,'count':len(yy),'prediction_flips':int((changed.argmax(1)!=pp.argmax(1)).sum()),
                                        'balanced_accuracy':scores(yy,changed)['balanced_accuracy'],
                                        'mean_probability_drift':float(np.abs(changed-pp).mean())})
                evaluation['ideal_gain_sensitivity']=sensitivity
        result={**info,'evaluation':evaluation};dump(md/'evaluation.json',result);results.append(result)
        if method=='raw':
            ref=json.loads((base/tag/'spectrum/evaluation.json').read_text(encoding='utf-8'))
            for test in ['same_surface_test','unseen_surface_test','unseen_surface_disjoint']:
                assert evaluation[test]==ref[test]
        print(f"EVAL {tag} {method} target={evaluation['unseen_surface_test']['balanced_accuracy']:.4f}",flush=True)
    dump(out/'summary.json',{'protocol':json.loads((out/'protocol_locked.json').read_text(encoding='utf-8')),
                            'experiments':results,'training_elapsed_seconds':frozen['training_elapsed_seconds']})
    report(out)


def report(out):
    out=Path(out);rows=json.loads((out/'summary.json').read_text(encoding='utf-8'))['experiments']
    text='# 频谱跨支撑面不稳定：归一化、增强和相对频带对照\n\n'
    text+='两个迁移方向、三个固定种子，36次CNN训练与12个简单类中心模型。原结构须精确复现上轮全部训练历史、权重及测试指标。训练只用源场地；所有模型和源场地选型冻结后才计算目标成绩。目标数据已经被历史研究查看，本轮属于回顾性开发证据，不是独立新数据盲测。\n\n'
    text+='## 主要发现与证据强度\n\n'
    text+='1. **幅值依赖是可修复因素，但不是唯一原因。** 原网络对理想幅值缩放高度敏感；归一化能消除这些数学缩放的影响，但单独均方根归一化在实际迁移中的结果仍不稳定。不能把零翻转等同于真实力度鲁棒。\n'
    for method in ['raw','rms','rms_log','band64_centroid']:
        vals={s:100*np.mean([x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rows if x['source']==s and x['method']==method]) for s in ['carpet','table']}
        text+=f"   - {LABELS[method]}：地毯→桌面{vals['carpet']:.2f}%，桌面→地毯{vals['table']:.2f}%。\n"
    text+='2. **频谱形状存在支撑相关变化。** 两根木头点3的频谱重心均移动约85～87Hz，相对频带分布的支撑间差异大于同一支撑面上的木头间差异；但木头之间的差异方向大体保留。说明区分类别的信息仍在，干扰也很强。表面、力度、采集顺序和接触状态尚未解耦，不能把这些变化全部因果归给支撑面。\n'
    text+='3. **输入动态范围和表示方式比取消步长更值得处理。** 归一化加对数压缩改善了地毯→桌面且保留反方向成绩；64频带相对能量的简单模型也有竞争力。这支持继续研究对幅值不敏感、限制强峰支配和降低拟合自由度的表示，不能据此认定某个单独机制是唯一解释。\n'
    text+='4. **数据覆盖和验证权重存在缺口。** 木头1点1地毯只有5条记录，源训练仅3条；地毯源验证组合的类别数为20:80。辅助等权验证对照与新源划分复核分别保存，不能混作第一阶段的独立盲测。\n\n'
    aux=out/'balanced_validation/summary.json'
    if aux.exists():
        a=json.loads(aux.read_text(encoding='utf-8'))
        text+=f"辅助对照：只将源验证损失改为两类等权，地毯→桌面平均提高{a['mean_delta_percentage_points']:.2f}个百分点。训练轨迹核验见balanced_validation/training_trajectory_verification.json；各种子训练损失与准确率逐轮完全相同，只有选中的轮次不同。说明类别不平衡影响了选轮次，但并未解释全部跨场地差距。该辅助规则写在主模型冻结后、读取主目标结果前，详见balanced_validation_timing_note.json。\n\n"
    if (out/'additional_source_splits/report.md').exists():
        text+='新源划分复核：见[额外三个源划分的完整结果](additional_source_splits/report.md)。64频带主候选由事先写下的最大化较差方向规则选出；最佳CNN是开发结果出来后追加的检查。目标原始记录和目标组合均未更换，因此它只检查源划分稳定性，不能称为独立目标确认。\n\n'
        extra=json.loads((out/'additional_source_splits/summary.json').read_text(encoding='utf-8'))['experiments']
        combined=rows+extra
        text+='## 六个源划分的最终比较\n\n| 方向 | 方法 | 原三划分均值 | 新三划分均值 | 六划分均值（范围） | 六划分不复用组均值 |\n|---|---|---:|---:|---:|---:|\n'
        for source in ['carpet','table']:
            for method in ['raw','rms_log','band64_centroid']:
                a=[r for r in rows if r['source']==source and r['method']==method]
                b=[r for r in extra if r['source']==source and r['method']==method]
                vals=[r['evaluation']['unseen_surface_test']['balanced_accuracy']*100 for r in a+b]
                first=np.mean([r['evaluation']['unseen_surface_test']['balanced_accuracy']*100 for r in a])
                second=np.mean([r['evaluation']['unseen_surface_test']['balanced_accuracy']*100 for r in b])
                dis=np.mean([r['evaluation']['unseen_surface_disjoint']['balanced_accuracy']*100 for r in a+b])
                text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {LABELS[method]} | {first:.2f}% | {second:.2f}% | {np.mean(vals):.2f}%（{min(vals):.2f}～{max(vals):.2f}） | {dis:.2f}% |\n"
        text+='\n建议优先保留均方根归一化＋对数压缩CNN作为改进网络候选，原有步长和参数量不变；若更重视较差迁移方向，64频带类中心是有竞争力的低复杂度候选。两种方法均非每个划分都胜出，改善的是平均表现及最差情况。尚未把等权验证与归一化＋对数组合试验，因此不能把两者收益相加。两条路线都须用新增、支撑交错且完整三点采集的数据作外部验收，本轮不替换固件。\n\n'
        stability_chart(out,combined)
    text+='## 描述性排查\n\n| 木头 | 敲击点 | 地毯/桌面记录数 | 桌面/地毯的均方根中位数比 | 相对频带能量分布差异 | 频谱重心变化（桌面减地毯，Hz） |\n|---|---:|---:|---:|---:|---:|\n'
    diag=json.loads((out/'signal_diagnosis.json').read_text(encoding='utf-8'))
    for r in diag['cross_surface_comparisons']:
        text+=f"| {r['wood']} | {r['point']} | {r['carpet_count']}/{r['table_count']} | {r['table_over_carpet_rms_median']:.2f} | {r['relative_band_power_total_variation']:.3f} | {r['table_minus_carpet_centroid_hz']:+.1f} |\n"
    text+='\n相对频带能量分布差异是两个归一化平均频带分布的总变差距离，0表示完全相同，1表示能量支持完全分离。这是分布的描述，未消除力度、顺序、接触状态等混杂，不能把全部差异因果归给支撑面。\n\n'
    text+='| 敲击点 | 地毯上两根木头的频带分布差异 | 桌面上两根木头的频带分布差异 | 两场地下类别差异方向余弦 |\n|---|---:|---:|---:|\n'
    for r in diag.get('wood_contrasts',[]):
        text+=f"| {r['point']} | {r['wood_separation_tv_carpet']:.3f} | {r['wood_separation_tv_table']:.3f} | {r['wood_contrast_direction_cosine_between_surfaces']:.3f} |\n"
    text+='\n类别差异方向余弦接近1表示两个场地下两类相对频带差异方向一致，接近0表示不一致，负值表示整体方向相反；它不是识别准确率。\n\n'
    text+='## 所有方法的目标成绩\n\n表中均为平衡准确率。组合复用原始敲击，不复用组三点中的每条敲击最多使用一次；三个种子也共享同一目标记录池。没有经质量闸门筛掉难例。\n\n| 方向 | 方法 | 同场地均值 | 跨场地均值（范围） | 配对变化，百分点 | 三次改善数 | 不复用组均值 |\n|---|---|---:|---:|---:|---:|---:|\n'
    csvrows=[]
    for source,title in [('carpet','地毯→桌面'),('table','桌面→地毯')]:
        base={x['seed']:x for x in rows if x['source']==source and x['method']=='raw'}
        for method in METHODS+CLASSICAL:
            rr=[x for x in rows if x['source']==source and x['method']==method]
            vals=[100*x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rr]
            deltas=[v-100*base[x['seed']]['evaluation']['unseen_surface_test']['balanced_accuracy'] for v,x in zip(vals,rr)]
            same=np.mean([100*x['evaluation']['same_surface_test']['balanced_accuracy'] for x in rr]);dis=np.mean([100*x['evaluation']['unseen_surface_disjoint']['balanced_accuracy'] for x in rr])
            text+=f"| {title} | {LABELS[method]} | {same:.2f}% | {np.mean(vals):.2f}%（{min(vals):.2f}～{max(vals):.2f}） | {np.mean(deltas):+.2f} | {sum(v>0 for v in deltas)}/3 | {dis:.2f}% |\n"
            for x,v,delta in zip(rr,vals,deltas):
                rec=x['evaluation']['unseen_surface_test']['per_class_recall']
                csvrows.append([source,x['target'],x['seed'],method,v,delta,rec[0]*100,rec[1]*100,x['evaluation']['unseen_surface_disjoint']['balanced_accuracy']*100])
    with (out/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['source','target','seed','method','target_balanced_accuracy_percent','delta_percentage_points','wood1_recall_percent','wood2_recall_percent','disjoint_balanced_accuracy_percent']);w.writerows(csvrows)
    text+='\n## 源场地验证选型能否迁移\n\n| 方向 | 种子 | 源验证选中CNN | 跨场地平衡准确率 | 源验证选中类中心 | 跨场地平衡准确率 |\n|---|---:|---|---:|---|---:|\n'
    for source,target in [('carpet','table'),('table','carpet')]:
        for seed in [42,123,2026]:
            sel=json.loads((out/f'{source}_to_{target}_seed{seed}'/'selection_locked.json').read_text(encoding='utf-8'))
            nn=next(x for x in rows if x['source']==source and x['seed']==seed and x['method']==sel['cnn_selected'])
            cl=next(x for x in rows if x['source']==source and x['seed']==seed and x['method']==sel['classical_selected'])
            text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {seed} | {LABELS[nn['method']]} | {nn['evaluation']['unseen_surface_test']['balanced_accuracy']*100:.2f}% | {LABELS[cl['method']]} | {cl['evaluation']['unseen_surface_test']['balanced_accuracy']*100:.2f}% |\n"
    text+='\n## 理想幅值扰动敏感性\n\n固定权重，对目标不复用组施加统一0.5倍、统一2倍、三点分别0.5/2/1.5倍的数学缩放。归一化的零翻转只说明抵消理想幅值缩放，不能证明真实力度变化、削顶和噪声都可解决。下表为每次划分三种缩放中最差翻转率的均值。\n\n| 方向 | 方法 | 最差预测翻转率均值 |\n|---|---|---:|\n'
    for source in ['carpet','table']:
        for method in METHODS+CLASSICAL:
            rr=[x for x in rows if x['source']==source and x['method']==method]
            rate=np.mean([max(y['prediction_flips']/y['count'] for y in x['evaluation']['ideal_gain_sensitivity']) for x in rr])
            text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {LABELS[method]} | {rate*100:.2f}% |\n"
    text+='\n## 每次划分和每根木头\n\n| 方向 | 种子 | 方法 | 目标平衡准确率 | 木头1召回 | 木头2召回 | 不复用组 |\n|---|---:|---|---:|---:|---:|---:|\n'
    for source,target,seed,method,v,delta,r1,r2,dis in csvrows:
        text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {seed} | {LABELS[method]} | {v:.2f}% | {r1:.2f}% | {r2:.2f}% | {dis:.2f}% |\n"
    text+='\n## 结论边界与复核\n\n归一化、对数压缩均在训练前重新适配并重训模型，不把分布突变的预处理直接塞给旧权重。所有统计仅拟合源训练池的独立原始敲击，端点统计合并以保持交换一致性。增强每批随机进行，不增加优化步数；形状扰动幅度预先固定，未按目标成绩调整。简单分类器只读取相对频带能量，不含时间衰减信息。\n\n源场地验证选型是可部署规则的回顾性检查；按目标成绩指出最有希望的方法属于开发分析，不构成独立确认。只有两根木头，表面与采集顺序混杂，木头1点1地毯仅5条；需要打乱支撑与敲击顺序、均衡各木头各点样本，并新增完整三点现场采集。原始数据、部署权重和固件未覆盖。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    chart(out,rows)
    print('REPORT '+str(out/'report.md'),flush=True)


def chart(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Microsoft YaHei'
    fig,axes=plt.subplots(2,1,figsize=(13,10),sharex=True)
    short=['原频谱','均方根\n归一化','频谱幅值和\n归一化','归一化\n＋对数','力度\n增强','归一化\n＋形状增强','16频带\n类中心','64频带\n类中心']
    for ax,source,title in zip(axes,['carpet','table'],['地毯训练 → 桌面测试','桌面训练 → 地毯测试']):
        for i,method in enumerate(METHODS+CLASSICAL):
            values=[100*x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rows if x['source']==source and x['method']==method]
            ax.bar(i,np.mean(values),color='#4b7893' if i<6 else '#55a18c',alpha=.8)
            ax.scatter(i+np.array([-.12,0,.12]),values,color='#202c36',s=24,zorder=3)
            ax.text(i,np.mean(values)+1,f'{np.mean(values):.1f}%',ha='center',fontsize=10)
        ax.set(title=title,ylim=(0,108),ylabel='目标平衡准确率（%）',xticks=range(8),xticklabels=short)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('频谱跨支撑面对照：柱为三个种子均值，点为各次结果')
    fig.text(.5,.01,'既有记录的回顾性开发对照；组合与种子不是独立新增采集。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.96));fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def stability_chart(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Microsoft YaHei'
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    methods=['raw','rms_log','band64_centroid']
    for ax,source,title in zip(axes,['carpet','table'],['地毯训练 → 桌面测试','桌面训练 → 地毯测试']):
        for i,method in enumerate(methods):
            rr=[x for x in rows if x['source']==source and x['method']==method]
            vals=[100*x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rr]
            ax.bar(i,np.mean(vals),color=['#597e9a','#cf9960','#5c9a86'][i],alpha=.85)
            old=[x['seed'] in [42,123,2026] for x in rr]
            offsets=np.linspace(-.16,.16,len(vals))
            for j,(v,is_old) in enumerate(zip(vals,old)):
                ax.scatter(i+offsets[j],v,color='#202c36',marker='o' if is_old else '^',s=27,zorder=3)
            ax.text(i,np.mean(vals)+1.5,f'{np.mean(vals):.2f}%',ha='center',fontsize=11)
        ax.set(title=title,ylim=(0,109),ylabel='目标平衡准确率（%）',xticks=range(3),xticklabels=['原频谱CNN','均方根归一化\n＋对数压缩CNN','64频带相对能量\n＋类中心分类器'])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('六个源划分复核：圆点为原三划分，三角为新增三划分')
    fig.text(.5,.01,'原始目标记录未更换；此图检查源划分稳定性，不是新增目标场地的盲测。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'six_split_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='luo_analysis/runs/20261019_cross_surface_v1');p.add_argument('--out',required=True);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.report_only: report(Path(a.out))
    elif a.evaluate_only: evaluate(a.base,a.out)
    else: train(a.base,a.out)
