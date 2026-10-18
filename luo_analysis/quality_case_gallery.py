"""Show raw traces and original PNG links with source-only quality error evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import config, preprocess
from .cross_surface import dump
from .quality import EXCLUDED
from .reporting import plotter
from .surface_audit import surface

PROFILES=['strict27','shape25','venue25']
PRIORITY=['WOOD02-E01-R00-P03-H006','WOOD02-E01-R00-P03-H033','WOOD02-E01-R00-P03-H038',
          'WOOD02-E01-R00-P01-H009','WOOD02-E01-R00-P01-H027','WOOD02-E01-R00-P01-H038',
          'WOOD02-E01-R00-P02-H036','WOOD02-E01-R00-P02-H037']


def run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    s=json.loads((source/'summary.json').read_text(encoding='utf-8'));c=config();records={};originals={};evidence={}
    for src in s['protocol']['sources']:
        path=Path(src['path']);payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==src['sha256']
        for line in payload.decode('utf-8-sig').splitlines():
            r=json.loads(line);sid=r['sample_id'];preprocess(r,c)
            records[sid]=r;originals[sid]=path.parent/'plots'/path.stem/(sid+'.png');assert originals[sid].is_file()
            evidence[sid]={p:[] for p in PROFILES}
    for exp in s['experiments']:
        d=source/f"{exp['source']}_to_{exp['target']}_seed{exp['seed']}"
        for profile in PROFILES:
            for partition in ['unseen_surface_normal','unseen_surface_abnormal']:
                for p in json.loads((d/(profile+'_'+partition+'.json')).read_text(encoding='utf-8')):
                    evidence[p['sample_id']][profile].append(dict(seed=exp['seed'],source=exp['source'],target=exp['target'],
                        rejected=p['retest'],score=p['score'],threshold=p['threshold']))
    counts={sid:{p:sum(q['rejected'] for q in rr) for p,rr in profiles.items()} for sid,profiles in evidence.items()}
    assert all(len(evidence[sid][p])==3 for sid in records for p in PROFILES)
    false_candidates=[sid for sid in sorted(records) if sid not in EXCLUDED and any(counts[sid].values())]
    selected=PRIORITY+sorted(set(false_candidates)-set(PRIORITY))
    assert set(selected)==EXCLUDED|set(false_candidates)
    assert len(false_candidates)==132 and len(selected)==137
    table=[]
    for sid,r in records.items():
        table.append(dict(sample_id=sid,reference_label='user_identified_abnormal' if sid in EXCLUDED else 'assumed_normal',
            surface=surface(r['stick_id'],r['impact_point_id'],r['strike_id']),
            hard_rejections=0,strict_rejections=counts[sid]['strict27'],shape_rejections=counts[sid]['shape25'],
            venue_rejections=counts[sid]['venue25'],test_runs_per_profile=3,original_png=str(originals[sid])))
    with (out/'all_300_decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(table[0]));w.writeheader();w.writerows(table)
    dump(out/'per_case_evidence.json',evidence)
    plt=plotter()
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']='Microsoft YaHei';plt.rcParams['axes.unicode_minus']=False

    def plot_cases(ids,path,columns):
        nrows=(len(ids)+columns-1)//columns
        fig,axes=plt.subplots(nrows,columns,figsize=(8.5*columns,3.1*nrows),squeeze=False)
        for ax,sid in zip(axes.flat,ids):
            r=records[sid];bad=sid in EXCLUDED;cc=counts[sid]
            ax.plot(np.arange(2400)/8,r['raw'],lw=.75,color='tab:red' if bad else 'tab:orange')
            ax.axvline(32,color='gray',ls='--',lw=.8)
            status='指定异常：硬检查漏拦；三档质量检查均拦截' if bad else '暂按正常：被要求重测，需人工复核'
            title=sid+'\n'+status+f"\n拦截次数/3：硬检查0 ｜ 严格{cc['strict27']} ｜ 形态{cc['shape25']} ｜ 宽松{cc['venue25']}"
            ax.set(title=title,xlabel='时间（毫秒）',ylabel='原始采样值',xlim=(0,300),ylim=(0,4095))
            ax.title.set_fontsize(9);ax.tick_params(labelsize=8);ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(ids):]:ax.axis('off')
        fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

    plot_cases(PRIORITY,out/'priority_8_waveforms.png',2)
    header='# 质量误拦／漏拦波形图册\n\n'
    header+='只统计本轮双向迁移中的陌生支撑面测试：每条记录在适用方向测试3次，种子42、123、2026。表格数字均为“要求重测次数/3”，不是木头编号分类错误次数。\n\n'
    header+='硬检查指现有协议、采集异常标记、削顶及弱信号检查；严格档含27特征；形态档去掉绝对幅值，25特征；宽松档在形态档阈值上乘既定1.5。原始图片标题anomalies: NONE记录的是采集程序异常标记，不是后来质量模型的结论。\n\n'
    header+='5条指定异常：硬检查全部漏拦，三档质量判定均3/3拦住。宽松档对暂按正常数据仅误拦3个不同编号；它们尚未重新人工验收，因此更准确地称为误拦候选。严格档误拦候选128条，形态档124条，两者及宽松档并集132条。\n\n'
    header+='## 首页8条\n\n![首页8条波形](priority_8_waveforms.png)\n\n'
    def table_text(ids):
        text='| 原始图片（点击查看完整上下两图） | 支撑面 | 当前标签 | 硬检查 | 严格档 | 形态档 | 宽松档 |\n|---|---|---|---:|---:|---:|---:|\n'
        for sid in ids:
            r=records[sid];cc=counts[sid];label='指定异常' if sid in EXCLUDED else '暂按正常'
            name=f"木头{int(r['stick_id'][-2:])}·点{r['impact_point_id']}·第{r['strike_id']:03}条"
            support='地毯' if surface(r['stick_id'],r['impact_point_id'],r['strike_id'])=='carpet' else '桌面'
            text+=f"| [{name}]({originals[sid].as_posix()}) | {support} | {label} | 0/3 | {cc['strict27']}/3 | {cc['shape25']}/3 | {cc['venue25']}/3 |\n"
        return text
    header+=table_text(PRIORITY)
    header+='\n对于“指定异常”，0/3表示漏拦，3/3表示正确拦截；对于“暂按正常”，大于0表示误拦候选。木头2点3第038条与指定异常的木头2点1第038条是两条不同记录。图册按原始数据重画捕获区间，点击表格链接可查看原始PNG及5秒预览。\n\n'
    dump(out/'summary.json',{'source_run':str(source),'profiles':PROFILES,'source_files_sha256':s['protocol']['sources'],
        'abnormal_ids':sorted(EXCLUDED),'priority_ids':PRIORITY,'normal_rejection_union':len(false_candidates),
        'normal_rejection_unique_by_profile':{p:sum(counts[sid][p]>0 for sid in false_candidates) for p in PROFILES},
        'normal_labels_not_changed':True,'hard_gate_abnormal_detected':0,'neural_wood_identity_not_evaluated_per_single_hit':True})
    for page,start in enumerate(range(0,len(selected),12),1):
        ids=selected[start:start+12];image=f'gallery_{page:02}.png';plot_cases(ids,out/image,3)
        header+=f'## 图册第{page}页\n\n![第{page}页]({image})\n\n'+table_text(ids)+'\n'
    (out/'gallery.md').write_text(header,encoding='utf-8')
    for sid in PRIORITY:
        print(json.dumps({'sample_id':sid,'surface':surface(records[sid]['stick_id'],records[sid]['impact_point_id'],records[sid]['strike_id']),
                          'label':'abnormal' if sid in EXCLUDED else 'assumed_normal','rejections':counts[sid], 'original_png':str(originals[sid])},ensure_ascii=False))
    print('Gallery: '+str(out/'gallery.md'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
