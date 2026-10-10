"""Plots, metrics and signal diagnostics; conclusions are gated by provenance."""
import csv
from pathlib import Path
import numpy as np
from .core import dump
from .network import signal_features


def plotter():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def metrics(y,pred,k=2):
    cm = np.zeros((k,k),dtype=int)
    for a,b in zip(y,pred):
        cm[a,b] += 1
    recall = np.divide(np.diag(cm),cm.sum(1),out=np.zeros(k,dtype=float),where=cm.sum(1)>0)
    present = cm.sum(1)>0
    return {'count':len(y),'accuracy':float((y==pred).mean()),
            'balanced_accuracy':float(recall[present].mean()),
            'per_class_recall':[float(recall[i]) if present[i] else None for i in range(k)],
            'confusion_matrix_rows_actual_columns_predicted':cm.tolist()}


def training_plots(history,results,c,out,mark):
    plt = plotter()
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for name in ['train','validation']:
        axes[0].plot([r['epoch'] for r in history],[r[name+'_loss'] for r in history],label=name)
        axes[1].plot([r['epoch'] for r in history],[r[name+'_accuracy'] for r in history],label=name)
    axes[0].set_ylabel('Cross-entropy'); axes[1].set_ylabel('Accuracy')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(mark); fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=150); plt.close(fig)
    cm = np.array(results['test']['cnn']['confusion_matrix_rows_actual_columns_predicted'])
    fig,ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set_xticks([0,1],c['classes']); ax.set_yticks([0,1],c['classes'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(mark+'\nHeld-out groups')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=150); plt.close(fig)


def diagnostic(record,c):
    raw = np.asarray(record['raw'],dtype=float)
    x = (raw-raw.mean())/2048
    fs = c['sample_rate_hz']
    power = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    power[0] = 0
    freq = np.fft.rfftfreq(len(x),1/fs)
    post = x[c['trigger_index']:]
    energy = post*post
    cumulative = np.cumsum(energy)
    t90 = float(np.searchsorted(cumulative,.9*cumulative[-1])/fs*1000) if cumulative[-1] else 0.
    rms_blocks = np.array([np.sqrt(np.mean(b*b)) for b in np.array_split(post,12)])
    times = (np.arange(12)+.5)*len(post)/12/fs
    slope = float(np.polyfit(times,np.log(np.maximum(rms_blocks,1e-10)),1)[0])
    return {'peak_to_peak_adc':float(np.ptp(raw)), 'peak_scaled':float(np.abs(x).max()),
            'rms_scaled':float(np.sqrt(np.mean(x*x))),
            'dominant_frequency_hz':float(freq[power.argmax()]),
            'spectral_centroid_hz':float(np.sum(freq*power)/max(power.sum(),1e-12)),
            'post_trigger_t90_ms':t90, 'log_rms_decay_slope_per_s':slope}


def analyze(records,c,out):
    out = Path(out)
    rows = []
    for a in records:
        rows.append({'sample_id':a['record']['sample_id'],'stick_id':a['record']['stick_id'],
                     'label':a['label'] or '', 'group':a['group'], 'source':a['source'],
                     'release_method':a['release_method'], **diagnostic(a['record'],c)})
    if rows:
        with (out/'signal_features.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    labels_ready = bool(records) and all(a['label'] in c['classes'] for a in records)
    real = bool(records) and all(a['source']=='real' for a in records)
    electrical = bool(records) and all(a['electrical_verified'] for a in records)
    grouped = {}
    positions = {a['record']['sample_id']:a['record']['impact_point_id'] for a in records}
    for row in rows:
        key = f"{row['label'] or 'UNLABELLED'}|{row['group']}|P{positions[row['sample_id']]}|{row['release_method']}"
        grouped.setdefault(key,[]).append(row)
    summary = {key:{name:float(np.mean([r[name] for r in rs])) for name in list(rows[0])[6:]} for key,rs in grouped.items()} if rows else {}
    # Normalized shape features intentionally remove amplitude for the confound check.
    diagnostics = {'status':'READY_FOR_REVIEW' if labels_ready and real and electrical else 'PENDING_REAL_LABELLED_ELECTRICALLY_VERIFIED_DATA',
                   'real_data':real,'labels_ready':labels_ready,'electrical_verified':electrical,
                   'group_means':summary,
                   'decay_note':'log RMS slope is a descriptive window metric, not a fitted physical damping constant',
                   'separability_conclusion':'PENDING human review and independent batch evaluation',
                   'sampling_actions':['keep release methods and impact positions balanced across classes',
                                       'collect independent remount sessions', 'have Dai verify electrical quality']}
    if records:
        plt = plotter()
        mark = 'REAL DATA - REVIEW REQUIRED' if real else 'DEMO / UNKNOWN SOURCE - NO REAL CONCLUSION'
        fig,axes = plt.subplots(2,2,figsize=(11,7))
        selected = []
        for name in c['classes']+[None]:
            hit = next((a for a in records if a['label']==name),None)
            if hit is not None:
                selected.append(hit)
        for a in selected:
            x = a['input'][:,0]
            tag = a['label'] or a['record']['stick_id']
            axes[0,0].plot(np.arange(1024)/16,x,label=tag,alpha=.7)
            power = np.abs(np.fft.rfft(x*np.hanning(1024)))**2
            axes[0,1].plot(np.fft.rfftfreq(1024,1/16000),power/max(power.sum(),1e-12),label=tag)
        axes[0,0].axvline(16,color='gray',ls='--'); axes[0,0].set(xlabel='Time (ms)',ylabel='TIME_V0_1')
        axes[0,1].set(xlabel='Frequency (Hz)',ylabel='Normalized spectral power')
        for name in c['classes']:
            rs = [r for r in rows if r['label']==name]
            if rs:
                axes[1,0].scatter([r['peak_to_peak_adc'] for r in rs],[r['dominant_frequency_hz'] for r in rs],label=name,alpha=.4,s=12)
                gs = sorted(set(r['group'] for r in rows))
                axes[1,1].scatter([gs.index(r['group']) for r in rs],[r['post_trigger_t90_ms'] for r in rs],label=name,alpha=.4,s=12)
        axes[1,0].set(xlabel='Peak-to-peak ADC (amplitude)',ylabel='Dominant frequency (Hz)')
        axes[1,1].set(xlabel='Remount group index',ylabel='Post-trigger T90 (ms)')
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]: ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle(mark); fig.tight_layout(); fig.savefig(out/'signal_analysis.png',dpi=150); plt.close(fig)
        x = np.stack([a['input'] for a in records])
        shape = signal_features(x,True)
        diagnostics['normalized_shape_mean_by_label'] = {label:shape[[i for i,a in enumerate(records) if a['label']==label]].mean(0).tolist()
            for label in c['classes'] if any(a['label']==label for a in records)}
    dump(out/'signal_report.json',diagnostics)
    text = f"# 10月5日早期信号分析\n\n状态：{diagnostics['status']}\n\n已生成时域、频域、窗口衰减描述指标、按装夹批次统计和振幅对照。\n\n"
    text += "真实可分性结论：待完成。没有实测证据时，不从模拟信号推断实心/空心的实际差异。\n\n"
    text += "检查 signal_analysis.png 和 signal_features.csv；训练报告会比较振幅单特征、去振幅形状特征与CNN。\n\n"
    text += "需李补充：两类木棍、多个独立装夹批次、均衡敲击位置、明确手动/机械释放。需戴确认：电气质量。\n"
    (out/'early_signal_report.md').write_text(text,encoding='utf-8')
    return diagnostics
