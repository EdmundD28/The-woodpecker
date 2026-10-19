"""Paired spectrum stride ablation; frozen source-only selection, no firmware edits."""
import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .core import preprocess
from .cross_surface import assemble, dump
from .network import CNN, windows, softmax
from .triplet import inputs, probabilities, scores

VARIANTS = {
    'stride222': ([2, 2, 2], None),
    'stride221': ([2, 2, 1], None),
    'stride111': ([1, 1, 1], None),
    'stride111_band151': ([1, 1, 1], 151),
}
LABELS = {'stride222':'原结构 2/2/2', 'stride221':'减少一次 2/2/1',
          'stride111':'无降采样 1/1/1', 'stride111_band151':'无降采样＋151频段（等参数）'}


class StudyCNN(CNN):
    """Reuse production convolutions/Adam, with optional position-preserving band head."""
    def __init__(self, c, weights=None):
        super().__init__(c)
        self.bands = c.get('study_bands')
        if self.bands:
            # Same dense initialization as baseline: all convolution tensor draws are identical.
            rng = np.random.default_rng(c['seed'])
            for i in range(len(self.layers)):
                rng.normal(size=self.p[f'w{i}'].shape)
            n = self.bands*self.layers[-1][0]
            self.p['wd'] = (rng.normal(size=(n,2))*np.sqrt(1/n)).astype(np.float32)
        if weights is not None:
            assert set(weights) == set(self.p)
            for k,v in weights.items():
                assert v.shape == self.p[k].shape and np.isfinite(v).all()
                self.p[k] = v.copy()

    def forward(self, x, cache=False):
        caches=[]
        for i,(_,k,s) in enumerate(self.layers):
            view,left=windows(x,k,s)
            z=(view.reshape(-1,k*x.shape[2]) @ self.p[f'w{i}'].reshape(k*x.shape[2],-1))
            z=z.reshape(x.shape[0],view.shape[1],-1)+self.p[f'b{i}']
            if cache: caches.append((x.shape,view,left,z>0))
            x=np.maximum(z,0)
        edges=np.linspace(0,x.shape[1],self.bands+1,dtype=int) if self.bands else None
        pooled=(np.stack([x[:,a:b].mean(1) for a,b in zip(edges[:-1],edges[1:])],axis=1)
                if self.bands else x).reshape(x.shape[0],-1)
        p=softmax(pooled @ self.p['wd']+self.p['bd'])
        return (p,(caches,x.shape,pooled,edges)) if cache else p

    def gradients(self,x,y):
        p,(caches,last_shape,pooled,edges)=self.forward(x,True)
        loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
        dz=p.copy(); dz[np.arange(len(y)),y]-=1; dz/=len(y)
        grad={'wd':pooled.T@dz,'bd':dz.sum(0)}
        dpool=dz @ self.p['wd'].T
        if self.bands:
            dpool=dpool.reshape(len(y),self.bands,last_shape[2])
            dx=np.empty(last_shape,dtype=np.float32)
            for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
                dx[:,a:b]=dpool[:,j:j+1]/(b-a)
        else: dx=dpool.reshape(last_shape)
        for i in reversed(range(len(self.layers))):
            shape,view,left,mask=caches[i]; _,k,s=self.layers[i]
            dz=dx*mask; flat=dz.reshape(-1,dz.shape[-1])
            grad[f'w{i}']=(view.reshape(-1,k*shape[2]).T@flat).reshape(self.p[f'w{i}'].shape)
            grad[f'b{i}']=dz.sum((0,1))
            dview=(flat@self.p[f'w{i}'].reshape(k*shape[2],-1).T).reshape(view.shape)
            dp=np.zeros((shape[0],(view.shape[1]-1)*s+k,shape[2]),dtype=np.float32)
            for j in range(k): dp[:,j:j+s*view.shape[1]:s]+=dview[:,:,j]
            dx=dp[:,left:left+shape[1]]
        return loss,grad


def resources(c,net):
    n=1201; cin=3; jump=1; receptive=1; macs=0; sizes=[]
    for cout,k,s in c['conv_layers']:
        receptive+=(k-1)*jump; jump*=s
        n=(n+s-1)//s; macs+=n*cout*k*cin; sizes.append(n*cout); cin=cout
    return {'parameters':sum(v.size for v in net.p.values()), 'last_positions':n,
            'conv_receptive_bins':receptive,'last_grid_spacing_hz':jump*8000/2400,
            'conv_dense_macs_one_direction':macs+net.p['wd'].size,
            'two_buffer_bytes_estimate':max(sizes)*8,'weight_bytes':sum(v.nbytes for v in net.p.values())}


def run(previous,out):
    previous=Path(previous); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    old=json.loads((previous/'summary.json').read_text(encoding='utf-8'))
    protocol={'base_run':str(previous.resolve()),'base_summary_sha256':hashlib.sha256((previous/'summary.json').read_bytes()).hexdigest(),
              'variants':VARIANTS,'seeds':[42,123,2026], 'directions':['carpet_to_table','table_to_carpet'],
              'rules':['exact existing split manifests and triplet pairings; unchanged exclusions',
                       'same spectrum, channels, kernels, Adam, learning rate, augmentation and epoch budget',
                       'same initial convolution weights per seed; native flatten heads have different sizes',
                       'band151 averages nonoverlapping contiguous bins after all stride-one convolutions; same parameter count as baseline',
                       'freeze all models before ANY target test; select candidate by source validation loss only',
                       'no tuning after inspecting target results; no firmware replacement',
                       'retrospective same two woods/day; surfaces confounded with collection order',
                       'stride changes receptive field and compute as well as resolution; causal attribution limited',
                       'combinations reuse hits; seeds reuse target pool; no independent-sample significance claim']}
    dump(out/'protocol_locked.json',protocol)
    records={}
    for src in old['protocol']['sources']:
        payload=Path(src['path']).read_bytes(); assert hashlib.sha256(payload).hexdigest()==src['sha256']
        records.update({r['sample_id']:r for r in map(json.loads,payload.decode('utf-8-sig').splitlines()) if r})
    start=time.monotonic(); frozen=[]
    # Train every architecture first. Target scores cannot influence any training or selection.
    for exp in old['experiments']:
        tag=f"{exp['source']}_to_{exp['target']}_seed{exp['seed']}"
        source=previous/tag; dest=out/tag; dest.mkdir()
        pairings=json.loads((source/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        for filename in ['identity_split.csv','triplet_sample_ids.json']:
            (dest/filename).write_bytes((source/filename).read_bytes())
        c=json.loads((source/'spectrum/config.json').read_text(encoding='utf-8'))
        cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
        assembled={name:assemble(pairings[name],records,cache) for name in ['train','validation']}
        xt,yt=assembled['train']; xv,yv=assembled['validation']
        tx=inputs(xt,c); vx=inputs(xv,c)
        tx=np.concatenate([tx,tx[:,:,::-1]]); yt=np.tile(yt,2)
        vx=np.concatenate([vx,vx[:,:,::-1]]); yv=np.tile(yv,2)
        initial=None; validations={}
        for name,(strides,bands) in VARIANTS.items():
            cc={**c,'conv_layers':[[co,k,s] for (co,k,_),s in zip(c['conv_layers'],strides)],'study_bands':bands}
            net=StudyCNN(cc); md=dest/name; md.mkdir()
            conv={k:v.copy() for k,v in net.p.items() if k not in ['wd','bd']}
            if initial is None: initial=conv
            else:
                for k in conv: np.testing.assert_array_equal(conv[k],initial[k])
            print('TRAIN '+tag+' '+name,flush=True)
            t=time.monotonic(); history=net.train(tx,yt,vx,yv,cc)
            dump(md/'config.json',cc); dump(md/'history.json',history)
            np.savez_compressed(md/'weights.npz',**net.p)
            val=scores(assembled['validation'][1],probabilities(net,inputs(assembled['validation'][0],cc)))
            info={'source':exp['source'],'target':exp['target'],'seed':exp['seed'],'variant':name,
                  'resources':resources(cc,net),'epochs':len(history),'best_epoch':min(history,key=lambda r:r['validation_loss'])['epoch'],
                  'training_seconds':time.monotonic()-t,'validation':val,
                  'selection_validation_loss':min(r['validation_loss'] for r in history),
                  'weights_sha256':hashlib.sha256((md/'weights.npz').read_bytes()).hexdigest()}
            dump(md/'frozen.json',info); frozen.append((tag,name,info))
            validations[name]=info['selection_validation_loss']
            print(f"FROZEN {tag} {name} val={validations[name]:.6f} seconds={info['training_seconds']:.1f}",flush=True)
        dump(dest/'selection_locked.json',{'criterion':'minimum source validation loss, endpoint augmentation; ties in declared variant order',
                                          'losses':validations,'selected':min(validations,key=validations.get)})
    dump(out/'all_models_frozen.json',{'models':[x[2] for x in frozen],'elapsed_seconds':time.monotonic()-start})
    evaluate(previous,out)


def evaluate(previous,out):
    previous=Path(previous);out=Path(out)
    locked=json.loads((out/'all_models_frozen.json').read_text(encoding='utf-8'))
    frozen=[(f"{r['source']}_to_{r['target']}_seed{r['seed']}",r['variant'],r) for r in locked['models']]
    protocol=json.loads((out/'protocol_locked.json').read_text(encoding='utf-8'))
    old=json.loads((previous/'summary.json').read_text(encoding='utf-8'))
    records={}
    for src in old['protocol']['sources']:
        payload=Path(src['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==src['sha256']
        records.update({r['sample_id']:r for r in map(json.loads,payload.decode('utf-8-sig').splitlines())})
    c=json.loads((out/frozen[0][0]/frozen[0][1]/'config.json').read_text(encoding='utf-8'))
    cache={sid:preprocess(r,c)[:,0] for sid,r in records.items()}
    results=[]
    for tag,name,info in frozen:
        dest=out/tag; md=dest/name
        cc=json.loads((md/'config.json').read_text(encoding='utf-8'))
        with np.load(md/'weights.npz') as archive: net=StudyCNN(cc,dict(archive))
        assert hashlib.sha256((md/'weights.npz').read_bytes()).hexdigest()==info['weights_sha256']
        pairings=json.loads((dest/'triplet_sample_ids.json').read_text(encoding='utf-8'))
        evaluation={}
        for test in ['validation','same_surface_test','unseen_surface_test','same_surface_disjoint','unseen_surface_disjoint']:
            xx,yy=assemble(pairings[test],records,cache); xx=inputs(xx,cc)
            pp=probabilities(net,xx,batch_size=16)
            np.testing.assert_array_equal(pp,probabilities(net,xx[:,:,::-1],batch_size=16))
            evaluation[test]=scores(yy,pp)
            np.savez_compressed(md/(test+'_predictions.npz'),probabilities=pp,labels=yy)
        info={**info,'evaluation':evaluation}
        if name=='stride222':
            ref=json.loads((previous/tag/'spectrum/evaluation.json').read_text(encoding='utf-8'))
            for key in ['same_surface_test','unseen_surface_test','unseen_surface_disjoint']:
                info.setdefault('previous_run_baseline_delta',{})[key]=evaluation[key]['balanced_accuracy']-ref[key]['balanced_accuracy']
        dump(md/'evaluation.json',info); results.append(info)
        print(f"EVAL {tag} {name} target={evaluation['unseen_surface_test']['balanced_accuracy']:.4f}",flush=True)
    dump(out/'summary.json',{'protocol':protocol,'experiments':results,'training_elapsed_seconds':locked['elapsed_seconds']})
    report(out)


def report(out):
    data=json.loads((out/'summary.json').read_text(encoding='utf-8')); rows=data['experiments']
    with (out/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['source','target','seed','variant','parameters','source_balanced_accuracy','target_balanced_accuracy','target_disjoint_balanced_accuracy','target_wood1_recall','target_wood2_recall','paired_delta_percentage_points','source_validation_loss'])
        for x in rows:
            e=x['evaluation']; base=next(r for r in rows if r['source']==x['source'] and r['seed']==x['seed'] and r['variant']=='stride222')
            target=e['unseen_surface_test']; delta=100*(target['balanced_accuracy']-base['evaluation']['unseen_surface_test']['balanced_accuracy'])
            writer.writerow([x['source'],x['target'],x['seed'],x['variant'],x['resources']['parameters'],e['same_surface_test']['balanced_accuracy'],target['balanced_accuracy'],e['unseen_surface_disjoint']['balanced_accuracy'],*target['per_class_recall'],delta,x['selection_validation_loss']])
    text='# 频谱卷积步长对照：桌面与地毯双向迁移\n\n'
    text+='四种预先固定结构，两个方向、三个种子，共24次训练。所有模型和源场地选型冻结后，才统一计算目标场地测试成绩。所有改进均与本轮同环境重训的原结构配对比较；不把上轮旧权重当作本轮基线。\n\n'
    text+='## 本轮结论\n\n'
    stable=[]
    for variant in list(VARIANTS)[1:]:
        deltas=[]
        for x in rows:
            if x['variant']!=variant: continue
            baseline=next(r for r in rows if r['source']==x['source'] and r['seed']==x['seed'] and r['variant']=='stride222')
            deltas.append(x['evaluation']['unseen_surface_test']['balanced_accuracy']-baseline['evaluation']['unseen_surface_test']['balanced_accuracy'])
        if all(v>0 for v in deltas): stable.append(variant)
    if not stable:
        text+='没有新结构在六次配对中全部改善，当前证据不支持将三次步长卷积视为跨场地失利的已证实主要瓶颈，也不支持直接替换部署模型。更细的频率表示并没有自动带来更强的跨支撑面能力。这个结果不能证明步长卷积完全不损失有用信息。\n\n'
    for source,label in [('carpet','地毯→桌面'),('table','桌面→地毯')]:
        rr=[x for x in rows if x['source']==source]
        means={v:100*np.mean([x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rr if x['variant']==v]) for v in VARIANTS}
        text+=f"- {label}：原结构{means['stride222']:.2f}%；完全取消降采样{means['stride111']:.2f}%（{means['stride111']-means['stride222']:+.2f}个百分点）；等参数晚压缩{means['stride111_band151']:.2f}%（{means['stride111_band151']-means['stride222']:+.2f}个百分点）。\n"
    text+='\n等参数版本地毯→桌面的均值改善须结合后面的三次配对和不复用组结果判断，不能单独当作稳定提升。源场地验证偏好的结构在目标场地也可能明显退步，说明同场地验证不足以预测跨场地能力。以下均为识别层成绩，未通过质量闸门筛掉难例后再计算。\n\n'
    text+='## 结构与成本\n\n'
    text+='| 结构 | 参数 | 末层位置数 | 卷积感受野（频点） | 单方向乘加 |\n|---|---:|---:|---:|---:|\n'
    for v in VARIANTS:
        r=next(x for x in rows if x['variant']==v)['resources']
        text+=f"| {LABELS[v]} | {r['parameters']} | {r['last_positions']} | {r['conv_receptive_bins']} | {r['conv_dense_macs_one_direction']} |\n"
    text+='\n末层位置间距不是输入频谱分辨率；输入始终是8000/2400≈3.333Hz。三次步长卷积并不等价于把原频谱简单改成26.667Hz分辨率。\n\n'
    text+='| 迁移方向 | 结构 | 源场地平衡准确率均值 | 目标平衡准确率均值（范围） | 相对原结构变化（百分点，均值） | 目标不复用组均值 |\n|---|---|---:|---:|---:|---:|\n'
    for source,target in [('carpet','table'),('table','carpet')]:
        label='地毯→桌面' if source=='carpet' else '桌面→地毯'
        for v in VARIANTS:
            rr=[x for x in rows if x['source']==source and x['variant']==v]
            vals=[x['evaluation']['unseen_surface_test']['balanced_accuracy']*100 for x in rr]
            base={x['seed']:x['evaluation']['unseen_surface_test']['balanced_accuracy']*100 for x in rows if x['source']==source and x['variant']=='stride222'}
            diff=[val-base[x['seed']] for val,x in zip(vals,rr)]
            same=np.mean([x['evaluation']['same_surface_test']['balanced_accuracy']*100 for x in rr])
            dis=np.mean([x['evaluation']['unseen_surface_disjoint']['balanced_accuracy']*100 for x in rr])
            text+=f"| {label} | {LABELS[v]} | {same:.2f}% | {np.mean(vals):.2f}%（{min(vals):.2f}～{max(vals):.2f}） | {np.mean(diff):+.2f} | {dis:.2f}% |\n"
    text+='\n## 按源场地验证损失选型的结果\n\n| 方向 | 种子 | 选中结构 | 目标平衡准确率 | 原结构 |\n|---|---:|---|---:|---:|\n'
    selections=[]
    for source,target in [('carpet','table'),('table','carpet')]:
        for seed in [42,123,2026]:
            tag=f'{source}_to_{target}_seed{seed}'
            sel=json.loads((out/tag/'selection_locked.json').read_text(encoding='utf-8'))['selected']
            chosen=next(x for x in rows if x['source']==source and x['seed']==seed and x['variant']==sel)
            baseline=next(x for x in rows if x['source']==source and x['seed']==seed and x['variant']=='stride222')
            cv=chosen['evaluation']['unseen_surface_test']['balanced_accuracy']*100; bv=baseline['evaluation']['unseen_surface_test']['balanced_accuracy']*100
            selections.append({'source':source,'seed':seed,'selected':sel,'target':cv,'baseline':bv})
            text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {seed} | {LABELS[sel]} | {cv:.2f}% | {bv:.2f}% |\n"
    dump(out/'source_selected_results.json',selections)
    text+='\n## 本轮原结构与上轮的复现差异\n\n本轮NumPy 2.3.5，上轮环境NumPy 2.5.3。额外使用旧环境、生产训练代码复算地毯源场地种子42：全部训练历史完全相同，权重最大差值为0，未使用目标测试数据。当前环境微小数值差异会累积为不同训练轨迹，故本轮比较以同环境、同实现的原结构重训为基线。以下为它相对前轮目标成绩的变化，不能算作架构收益。详见previous_runtime_probe.json。\n\n'
    for x in rows:
        if x['variant']=='stride222':
            delta=x['previous_run_baseline_delta']['unseen_surface_test']*100
            text+=f"- {x['source']}→{x['target']}，种子{x['seed']}：{delta:+.2f}个百分点。\n"
    text+='\n## 配对变化\n\n| 方向 | 结构 | 三个种子相对原结构变化（百分点） | 改善次数 |\n|---|---|---|---:|\n'
    for source in ['carpet','table']:
        baseline={x['seed']:x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rows if x['source']==source and x['variant']=='stride222'}
        for variant in list(VARIANTS)[1:]:
            changes=[100*(x['evaluation']['unseen_surface_test']['balanced_accuracy']-baseline[x['seed']]) for x in rows if x['source']==source and x['variant']==variant]
            text+=f"| {'地毯→桌面' if source=='carpet' else '桌面→地毯'} | {LABELS[variant]} | {', '.join(f'{v:+.2f}' for v in changes)} | {sum(v>0 for v in changes)}/3 |\n"
    text+='\n## 每次划分与每根木头\n\n| 方向 | 种子 | 结构 | 目标平衡准确率 | 木头1召回率 | 木头2召回率 | 不复用组平衡准确率 |\n|---|---:|---|---:|---:|---:|---:|\n'
    for x in rows:
        e=x['evaluation']['unseen_surface_test']; d=x['evaluation']['unseen_surface_disjoint']; rec=e['per_class_recall']
        text+=f"| {x['source']}→{x['target']} | {x['seed']} | {LABELS[x['variant']]} | {e['balanced_accuracy']*100:.2f}% | {rec[0]*100:.2f}% | {rec[1]*100:.2f}% | {d['balanced_accuracy']*100:.2f}% |\n"
    text+='\n## 解释边界\n\n无降采样展平同时增加输出层参数；等参数版本先在完整频率轴执行三层卷积，再按151个连续频段平均，保留频段位置。它检验的是“晚压缩是否比早压缩更好”，不是完全不压缩。步长变化还把感受野从25频点缩到15频点，不能将全部差异只归因于频率细节。\n\n目标组合重复使用原始敲击，三个种子也共享目标记录；不把组合数当独立样本，不做虚假的显著性声明。不复用测试组更少，尤其地毯木头1只有5组。表面与采集顺序混杂，仅两根木头、同一天。该研究是回顾性对照，仍需独立新采集确认。没有修改部署模型或固件。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    chart(out,rows)
    print(text,flush=True)


def chart(out,rows):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print('Plotting library absent; numeric report is complete. Render chart with project runtime.',flush=True)
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Microsoft YaHei'
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,source,title in zip(axes,['carpet','table'],['地毯训练 → 桌面测试','桌面训练 → 地毯测试']):
        for j,variant in enumerate(VARIANTS):
            vals=[100*x['evaluation']['unseen_surface_test']['balanced_accuracy'] for x in rows if x['source']==source and x['variant']==variant]
            ax.bar(j,np.mean(vals),color=['#456b91','#579d90','#cc8150','#8b78ac'][j],alpha=.85)
            ax.scatter(j+np.array([-.12,0,.12]),vals,color='#202c36',s=24,zorder=3)
            ax.text(j,np.mean(vals)+1,f'{np.mean(vals):.1f}%',ha='center',fontsize=10)
        ax.set(title=title,xticks=range(4),xticklabels=['原结构\n2/2/2','减少一次\n2/2/1','无降采样\n1/1/1','无降采样\n等参数频段头'],ylim=(0,108),ylabel='目标场地平衡准确率（%）')
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('频谱步长对照：柱为三个种子均值，点为各次结果')
    fig.text(.5,.01,'相同两根木头、既有敲击重复划分；组合与种子并非独立采集样本。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95)); fig.savefig(out/'comparison.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',default='cnn_analysis/runs/20261019_cross_surface_v1');p.add_argument('--out',required=True);p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    if args.evaluate_only: evaluate(args.base,args.out)
    else: run(args.base,args.out)
