"""Render all predeclared support-surface experiments without choosing a winner."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .reporting import diagnostic, plotter
from .core import config
from .quality import EXCLUDED
from .surface_audit import surface

LABEL = {'carpet':'地毯','table':'桌面','time':'时域','spectrum':'频谱','strict27':'严格27特征','shape25':'形态25特征','venue25':'当前宽松档'}


def interval(values, percent=True):
    x=np.asarray(values,dtype=float)
    return f'{x.mean():.1%}（{x.min():.1%}～{x.max():.1%}）' if percent else f'{x.mean():.1f}（{x.min():.0f}～{x.max():.0f}）'


def write_csv(path, rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)


def run(out):
    out=Path(out); summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    results=summary['experiments']; assert len(results)==6
    cnn_rows=[]; quality_rows=[]; single_rows=[]; bad_rows=[]; pipeline_rows=[]; hit_errors=[]; review={}
    for r in results:
        direction=LABEL[r['source']]+'→'+LABEL[r['target']]
        for mode,ev in r['cnn'].items():
            for test in ['same_surface_test','unseen_surface_test','same_surface_disjoint','unseen_surface_disjoint']:
                m=ev[test]
                cnn_rows.append(dict(direction=direction,seed=r['seed'],model=mode,test=test,**{k:m[k] for k in ['count','accuracy','balanced_accuracy']},
                                     wood1_recall=m['per_class_recall'][0],wood2_recall=m['per_class_recall'][1],epochs=ev['epochs_run'],best_epoch=ev['best_epoch'],parameters=ev['parameters']))
            for profile,p in ev['pipeline'].items():pipeline_rows.append(dict(direction=direction,seed=r['seed'],mode=mode,profile=profile,**p))
            dest=out/f"{r['source']}_to_{r['target']}_seed{r['seed']}"
            grouped={}
            for p in json.loads((dest/mode/'unseen_surface_test_predictions.json').read_text(encoding='utf-8')):
                for sid in p['sample_ids']:
                    g=grouped.setdefault(sid,{'combinations':0,'errors':0});g['combinations']+=1;g['errors']+=int(p['actual']!=p['prediction'])
            for sid,g in grouped.items():hit_errors.append(dict(direction=direction,seed=r['seed'],mode=mode,sample_id=sid,**g,conditional_error_rate=g['errors']/g['combinations']))
        for profile,ev in r['quality'].items():
            for test in ['same_surface_normal','unseen_surface_normal','same_surface_abnormal','unseen_surface_abnormal']:
                for group,m in ev[test].items():quality_rows.append(dict(direction=direction,seed=r['seed'],profile=profile,test=test,group=group,**m))
            dest=out/f"{r['source']}_to_{r['target']}_seed{r['seed']}"
            for test in ['same_surface_abnormal','unseen_surface_abnormal']:
                for p in json.loads((dest/(profile+'_'+test+'.json')).read_text(encoding='utf-8')):
                    bad_rows.append(dict(direction=direction,seed=r['seed'],profile=profile,test=test,sample_id=p['sample_id'],retest=p['retest'],score=p['score'],threshold=p['threshold']))
            if profile=='venue25':
                for p in json.loads((dest/(profile+'_unseen_surface_normal.json')).read_text(encoding='utf-8')):
                    if p['retest']:review.setdefault(p['sample_id'],[]).append(r['seed'])
        for model,tests in r['single_point'].items():
            for test,m in tests.items():single_rows.append(dict(direction=direction,seed=r['seed'],model=model,test=test,count=m['count'],balanced_accuracy=m['balanced_accuracy']))
    for name,rows in [('identity_metrics.csv',cnn_rows),('quality_metrics.csv',quality_rows),('single_point_metrics.csv',single_rows),('abnormal_decisions.csv',bad_rows),('pipeline_metrics.csv',pipeline_rows),('hit_conditional_errors.csv',hit_errors)]:write_csv(out/name,rows)
    text='# 桌面与地毯双向迁移实验\n\n'
    text+='只用一种支撑面的记录训练和设定阈值，再测试另一种支撑面。两根已知木头、同一天已有记录；不是新木头、新日期或所有陌生场地的验证。所有结果均报告，没有根据目标场地选最优种子、模式或阈值。\n\n'
    text+='本轮结论：严格质量检查跨支撑面时误拦明显增多，当前宽松档将陌生场地正常误拦降到0.6%～1.6%，且六次实验均拦住同样的5条指定异常。三点识别仍存在方向差异和训练划分波动；不能把质量检查通过等同于识别稳定。简单三点形态对照也已达到约97%，暂没有证据说明必须先加大网络。\n\n'
    text+='这是对既有数据的回顾性重新划分，数据此前已用于混合场地实验；本轮新模型只接触源场地，不是全新独立采集的盲测。\n\n'
    text+='## 数据和事先固定的规则\n\n'
    text+='正常参考共295条：桌面170条、地毯125条；其余5条指定异常完全排除出识别训练、验证及质量拟合、校准。异常中桌面仅1条、地毯4条，异常检出率必须连同分母阅读。其余记录暂按正常处理，尚未逐条人工验收。\n\n'
    text+='三个固定种子42、123、2026；源场地内部按木头和位置分别划分。识别用60%训练、20%验证、余下测试，模型只按源场地验证损失选训练轮次；训练最多120轮。质量用40%拟合、向上取整40%校准、余下测试，保证中点至少19条独立校准记录，能够计算有限的5%校准阈值。这个目标误拦率依赖同分布条件，跨场地不作5%保证。\n\n'
    text+='识别训练每根木头120个三点组合，另做反向增强；每组严格是一根木头、同一支撑面、点1/2/3各一条。推理平均123与321输出，全部测试逐组核实两方向结果完全相同。两端互换不等于允许把中点换到端点。每方向训练时，另一支撑面完全不参与参数、归一化尺度、阈值和训练轮次选择。\n\n'
    text+='木头1点1仅5条地毯记录：识别划为3训练、1验证、1同场地测试。'
    for source in ['carpet','table']:
        r=next(r for r in results if r['source']==source); counts=r['identity_counts']
        text+=f"{LABEL[source]}→{LABEL[r['target']]}识别{counts['train']}训练、{counts['validation']}验证、{counts['test']}同场地测试；"
    text+='质量划分见各次清单。\n\n'
    text+='## 三点识别：三次结果的均值与范围\n\n| 训练→陌生测试 | 网络 | 同场地平衡准确率 | 陌生场地平衡准确率 | 陌生场地不重复组三次平衡准确率 |\n|---|---|---:|---:|---:|\n'
    for source in ['carpet','table']:
        rr=[r for r in results if r['source']==source]
        for mode in ['time','spectrum']:
            text+=f"| {LABEL[source]}→{LABEL[rr[0]['target']]} | {LABEL[mode]} | {interval([r['cnn'][mode]['same_surface_test']['balanced_accuracy'] for r in rr])} | {interval([r['cnn'][mode]['unseen_surface_test']['balanced_accuracy'] for r in rr])} | {interval([r['cnn'][mode]['unseen_surface_disjoint']['balanced_accuracy'] for r in rr])} |\n"
    text+='\n平衡准确率为两根木头召回率的平均，避免不重复组数量差异掩盖某根木头。陌生场地随机测试每根木头1000组，共2000组；它们重复使用170或125条原始记录，不能当作2000次独立采集。不重复组测试每条记录最多使用一次：桌面49组（木头1为25、木头2为24），地毯28组（木头1为5、木头2为23）；三个种子重复测试相同原始池，也不是三倍独立样本。\n\n'
    text+='时域模型2514参数，频谱模型12114参数。频谱模型更大不保证跨场地更好；本实验不根据目标成绩替换固件。\n\n'
    text+='### 陌生场地：两根木头分别的召回率\n\n| 方向 | 网络 | 木头1 | 木头2 |\n|---|---|---:|---:|\n'
    for source in ['carpet','table']:
        rr=[r for r in results if r['source']==source]
        for mode in ['time','spectrum']:
            text+=f"| {LABEL[source]}→{LABEL[rr[0]['target']]} | {LABEL[mode]} | {interval([r['cnn'][mode]['unseen_surface_test']['per_class_recall'][0] for r in rr])} | {interval([r['cnn'][mode]['unseen_surface_test']['per_class_recall'][1] for r in rr])} |\n"
    text+='\n## 质量检查：是否会误拦正常敲击\n\n严格27特征包含绝对幅值；形态25特征去掉峰峰值与均方根幅值，但保留频谱及能量时间分布；当前宽松档在形态25特征的源场地校准阈值上乘固定1.5。1.5来自此前固件设置，本实验未用目标数据重新调整。\n\n'
    text+='| 方向 | 档位 | 同场地正常误拦率 | 陌生场地正常误拦率 | 陌生场地误拦数范围 | 陌生场地异常检出数（三次） |\n|---|---|---:|---:|---:|---|\n'
    for source in ['carpet','table']:
        rr=[r for r in results if r['source']==source]
        for profile in ['strict27','shape25','venue25']:
            vals=[r['quality'][profile] for r in rr]
            bad=', '.join(f"{v['unseen_surface_abnormal']['all']['retest']}/{v['unseen_surface_abnormal']['all']['count']}" for v in vals)
            counts=[v['unseen_surface_normal']['all']['retest'] for v in vals]; n=vals[0]['unseen_surface_normal']['all']['count']
            text+=f"| {LABEL[source]}→{LABEL[rr[0]['target']]} | {LABEL[profile]} | {interval([v['same_surface_normal']['all']['retest_rate'] for v in vals])} | {interval([v['unseen_surface_normal']['all']['retest_rate'] for v in vals])} | {min(counts)}～{max(counts)}/{n} | {bad} |\n"
    text+='\n每条指定异常逐次分数、阈值和拦截结果保存在abnormal_decisions.csv；正常误拦的木头、位置分布保存在quality_metrics.csv以及每次预测文件中。指定异常之前已经被查看过，所以仍属回顾证据，不能等同于新异常类型的检出率。\n\n'
    text+='当前宽松档在陌生场地上要求重测的假定正常记录如下。它们仍保留在正常测试池，未根据模型输出擅自改标签；需要结合采集经过或人工复核。图见normal_retest_review.png。\n\n'
    for sid,seeds in sorted(review.items()):text+=f"- {sid}：{len(seeds)}/3次划分要求重测。\n"
    text+='\n木头2点3第6条在触发附近响应较弱、后段出现更强振动，外观与部分指定误采相似，值得优先复核；这只是图形观察，不作为改标签或重算成绩的理由。这里的点3第38条与最初指定的点1第38条是不同记录。\n\n'
    text+='## 完整流程：三点都过质量关以后\n\n下表采用当前时域识别网络及当前宽松质量档。通过率是无需重测即可进入识别的三点组合比例；通过后准确率只看被放行的组合，同时列出“直接放行且判对”的总体比例，避免拒绝大量数据后准确率看起来虚高。未模拟重测，不能推算实际等待次数。\n\n| 方向 | 三点全部放行率 | 放行后总体准确率 | 直接放行且判对占全部组合 |\n|---|---:|---:|---:|\n'
    for source in ['carpet','table']:
        rr=[r for r in results if r['source']==source]; pp=[r['cnn']['time']['pipeline']['venue25'] for r in rr]
        text+=f"| {LABEL[source]}→{LABEL[rr[0]['target']]} | {interval([p['coverage'] for p in pp])} | {interval([p['accepted_accuracy'] for p in pp if p['accepted_accuracy'] is not None])} | {interval([p['correct_and_accepted_rate'] for p in pp])} |\n"
    text+='\n放行后的准确率按实际放行组合计算，不是平衡准确率，且组合之间共享原始记录。\n\n'
    text+='## 排查：仅靠幅值、仅靠形态，以及单点\n\n以下为训练集拟合的简单标准化最近类中心；用途是排查信息来源，不参与选神经网络或阈值。三点形态特征使用相对频带能量及时间能量分布，去掉绝对幅值；幅值基线仅使用峰峰值和均方根。\n\n| 方向 | 三点仅幅值 | 三点仅形态 |\n|---|---:|---:|\n'
    for source in ['carpet','table']:
        rr=[r for r in results if r['source']==source]
        text+=f"| {LABEL[source]}→{LABEL[rr[0]['target']]} | {interval([r['baselines']['amplitude_only']['unseen_surface_test']['balanced_accuracy'] for r in rr])} | {interval([r['baselines']['shape_without_amplitude']['unseen_surface_test']['balanced_accuracy'] for r in rr])} |\n"
    text+='\n| 方向 | 单点位置 | 只看形态：同场地 | 只看形态：陌生场地 | 只看幅值：陌生场地 |\n|---|---|---:|---:|---:|\n'
    for source in ['carpet','table']:
        rr=[r for r in results if r['source']==source]
        for p in [1,2,3]:
            text+=f"| {LABEL[source]}→{LABEL[rr[0]['target']]} | 点{p} | {interval([r['single_point'][f'P{p}:shape']['same_surface_test']['balanced_accuracy'] for r in rr])} | {interval([r['single_point'][f'P{p}:shape']['unseen_surface_test']['balanced_accuracy'] for r in rr])} | {interval([r['single_point'][f'P{p}:amplitude']['unseen_surface_test']['balanced_accuracy'] for r in rr])} |\n"
    text+='\n## 能证明什么，尚不能证明什么\n\n这些结果直接检验了“现有地毯记录训练→现有桌面记录”和反向迁移，证明或否定的是这两组数据之间的迁移能力。表面标签与连续采集时间段重合，也可能同时混入敲击力度、接触或传感器状态变化，不能把全部差异严格归因于地毯本身。数据只涉及两根木头；三个位置的记录不是一次连续展示采集，因此随机配组三点仍不能替代真实展示试验。\n\n'
    text+='去掉绝对幅值仍保留受支撑方式影响的频谱和衰减形态，所以不能预设“幅值归一化即可解决换场地”。质量检查放行也不保证木头识别正确；两者必须分别验收。当前宽松档若表现好，也只覆盖已见的桌面／地毯迁移，不能保证任意场地。5条异常数量太少，不支持用百分比承诺所有误采都会被拦住。\n\n'
    text+='下一步独立验证应保持现有模型冻结，在展示场地重新采三点完整流程，记录支撑方式、朝向、力度、是否要求重测和最终木头编号；每个条件分开统计。若再根据这批跨场地结果改模型或阈值，这批数据就变成开发依据，需要新增现场记录作为最终验收。\n\n'
    text+='## 可复核文件\n\nprotocol_locked.json记录预先固定的规则与源文件散列；每方向、每种子保存两套独立原始记录划分、全部三点组合编号、训练历史、冻结权重、逐条质量分数和逐组三点预测。summary.json汇总全部结果；CSV便于筛选。原始采集文件、已部署权重和固件均未覆盖。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    # Physical descriptors are descriptive only, never fitted to improve target performance.
    descriptors=[]; c=config(); raw_records={}
    for src in summary['protocol']['sources']:
        payload=Path(src['path']).read_bytes(); assert hashlib.sha256(payload).hexdigest()==src['sha256']
        for line in payload.decode('utf-8-sig').splitlines():
            r=json.loads(line)
            raw_records[r['sample_id']]=r
            if r['sample_id'] in EXCLUDED:continue
            descriptors.append(dict(sample_id=r['sample_id'],wood=r['stick_id'],point=r['impact_point_id'],surface=surface(r['stick_id'],r['impact_point_id'],r['strike_id']),**diagnostic(r,c)))
    write_csv(out/'normal_signal_descriptors.csv',descriptors)
    plt=plotter()
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']='Microsoft YaHei'; plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(len(review),1,figsize=(12,3*len(review)),squeeze=False)
    for ax,(sid,seeds) in zip(axes[:,0],sorted(review.items())):
        r=raw_records[sid]; raw=np.asarray(r['raw']);ax.plot(np.arange(len(raw))/8,raw,color='tab:orange',lw=.8)
        ax.axvline(32,color='red',ls='--');ax.set(title=f"{sid}：{len(seeds)}/3次要求重测，原标签暂按正常",xlabel='时间（毫秒）',ylabel='原始采样值');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'normal_retest_review.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,source in enumerate(['carpet','table']):
        rr=[r for r in results if r['source']==source]; ax=axes[i]
        for j,mode in enumerate(['time','spectrum']):
            same=[r['cnn'][mode]['same_surface_test']['balanced_accuracy']*100 for r in rr]
            target=[r['cnn'][mode]['unseen_surface_test']['balanced_accuracy']*100 for r in rr]
            for a,b in zip(same,target):ax.plot([j-.15,j+.15],[a,b],alpha=.45,color=['tab:blue','tab:orange'][j])
            ax.scatter([j-.15]*3,same,marker='o',color=['tab:blue','tab:orange'][j],label=LABEL[mode]+'同场地')
            ax.scatter([j+.15]*3,target,marker='x',s=65,color=['tab:blue','tab:orange'][j],label=LABEL[mode]+'陌生场地')
        ax.set(title=LABEL[source]+'训练 → '+LABEL[rr[0]['target']]+'测试',ylim=(0,105),xticks=[0,1],xticklabels=['时域2514参数','频谱12114参数'],ylabel='平衡准确率（%）')
        ax.axhline(50,ls='--',color='gray',label='随机猜测'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('同场地与陌生支撑面：每个点代表一次固定划分'); fig.tight_layout(); fig.savefig(out/'identity_transfer.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,source in zip(axes,['carpet','table']):
        rr=[r for r in results if r['source']==source]
        for j,profile in enumerate(['strict27','shape25','venue25']):
            yy=[r['quality'][profile]['unseen_surface_normal']['all']['retest_rate']*100 for r in rr]
            ax.scatter([j-.07,j,j+.07],yy,color=['tab:red','tab:orange','tab:green'][j])
        ax.set(title=LABEL[source]+'训练 → '+LABEL[rr[0]['target']]+'正常记录',xticks=[0,1,2],xticklabels=['严格27特征','形态25特征','当前宽松档'],ylabel='正常记录误拦率（%）',ylim=(-1,101))
        ax.axhline(5,ls='--',color='gray',label='源场地校准目标5%'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('陌生支撑面的质量检查：每个点代表一次固定划分'); fig.tight_layout(); fig.savefig(out/'quality_transfer.png',dpi=160); plt.close(fig)
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
