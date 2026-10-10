import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .core import ROOT, config, dump, load, preprocess, split, dataset_hash, model_input, input_contract
from .network import CNN, signal_features, fit_centroid, centroid_predict
from .reporting import analyze, metrics, training_plots
from .export import export
from pc_collector.protocol import build_sample_id


def new_output(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=False)
    return out


def prepare(paths,c,manifest,out):
    records,rejected,duplicates = load(paths,c,manifest)
    dump(out/'data_audit.json',{'accepted':len(records),'rejected':rejected,'duplicate_copies_skipped':duplicates,
         'unlabelled':sum(not a['label'] for a in records),
         'unknown_source':sum(a['source']=='unknown' for a in records),
         'dataset_sha256':dataset_hash(records)})
    dump(out/'config_snapshot.json',c)
    if records:
        time_inputs=np.stack([a['input'] for a in records])
        np.savez_compressed(out/'model_inputs.npz', inputs=time_inputs,
                            spectrum_inputs=model_input(time_inputs,{**c,'input_mode':'spectrum'}),
                            sample_ids=np.array([a['record']['sample_id'] for a in records]),
                            labels=np.array([a['label'] or '' for a in records]),
                            groups=np.array([a['group'] for a in records]),
                            spectrum_preprocessing_version=np.array('SPECTRUM_V0_1'),
                            preprocessing_version=np.array('TIME_V0_1'))
    analyze(records,c,out)
    return records


def train(records,c,out,demo=False):
    reason = None
    if not records:
        reason = 'NO_ACCEPTED_DATA'
    elif any(a['label'] not in c['classes'] for a in records):
        reason = 'LABEL_MAPPING_MISSING'
    elif demo and any(a['source']!='synthetic' for a in records):
        reason = 'DEMO_REQUIRES_SYNTHETIC_ONLY'
    elif not demo and any(a['source']!='real' for a in records):
        reason = 'CONFIRMED_REAL_PROVENANCE_REQUIRED'
    if reason is None:
        try:
            ids,partitions = split(records,c)
        except ValueError as e:
            reason = str(e)
    if reason:
        dump(out/'training_status.json',{'status':'PENDING','reason':reason,'model_exported':False})
        print('Training deferred: ' + reason)
        return
    time_x = np.stack([a['input'] for a in records])
    x = model_input(time_x,c)
    y = np.array([c['classes'].index(a['label']) for a in records])
    cnn = CNN(c)
    tr,va = ids['train'],ids['validation']
    history = cnn.train(x[tr],y[tr],x[va],y[va],c)
    baseline_sets = {'simple_signal_baseline':signal_features(time_x),
                     'amplitude_only':np.ptp(time_x,axis=1),
                     'shape_without_amplitude':signal_features(time_x,True)}
    baselines = {name:fit_centroid(f[tr],y[tr]) for name,f in baseline_sets.items()}
    results = {}
    all_probs = cnn.forward(x)
    errors = []
    for name,ix in ids.items():
        pred = all_probs[ix].argmax(1)
        results[name] = {'cnn':metrics(y[ix],pred)}
        for b,f in baseline_sets.items():
            results[name][b] = metrics(y[ix],centroid_predict(f[ix],baselines[b]))
        # Test subgroups help expose position, release-method and remount dependence.
        results[name]['subgroups'] = {}
        for kind,key in [('remount',lambda a:a['group']),('position',lambda a:str(a['record']['impact_point_id'])),
                         ('release',lambda a:a['release_method']),('stick',lambda a:a['record']['stick_id'])]:
            for value in sorted({key(records[i]) for i in ix}):
                subset = np.array([i for i in ix if key(records[i])==value])
                results[name]['subgroups'][kind+':'+value] = metrics(y[subset],all_probs[subset].argmax(1))
        if name == 'test':
            for i in ix:
                if all_probs[i].argmax() != y[i]:
                    errors.append({'sample_id':records[i]['record']['sample_id'], 'group':records[i]['group'],
                                   'actual':records[i]['label'],'predicted':c['classes'][int(all_probs[i].argmax())],
                                   'scores':all_probs[i].tolist(),'file':records[i]['file'],
                                   'line':records[i]['line'],'review_reason':'PENDING waveform/acquisition review'})
    weights = {n:a.tolist() for n,a in cnn.p.items()}
    identity = {'weights':weights, 'config':c, 'dataset_sha256':dataset_hash(records)}
    token = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    version = ('DEMO_ONLY-' if demo else 'REAL_CANDIDATE-')+token
    model = {'format_version':'WOOD_CNN_NUMPY_V1','model_version':version,'config':c,'weights':weights,
             'input_contract':input_contract(c),
             'dataset_sha256':dataset_hash(records),'demo_only':demo,
             'deployment_status':'REFERENCE_EXPORTED_ESP32_NOT_VERIFIED'}
    dump(out/'model.json',model)
    np.savez_compressed(out/'model_weights.npz',**cnn.p)
    dump(out/'baseline_models.json',baselines)
    dump(out/'history.json',history)
    dump(out/'error_samples.json',errors)
    dump(out/'split_groups.json',partitions)
    electrical = all(a['electrical_verified'] for a in records)
    status = 'DEMO_ONLY' if demo else 'REAL_CANDIDATE_REQUIRES_REVIEW'
    report = {'status':status,'model_version':version,'input_mode':c.get('input_mode','time'),
              'input_contract':input_contract(c),'dataset_sha256':model['dataset_sha256'],
              'electrical_verified':electrical,'best_epoch':min(history,key=lambda r:r['validation_loss'])['epoch'],
              'classes':c['classes'],'results':results,
              'warning':'Synthetic metrics only check code execution' if demo else 'Scores are not calibrated; review errors, sampling confounds and subgroup counts',
              'split_note':'Conservative global E/R groups, never random hits. Test set never used for checkpoint selection.',
              'confidence_threshold':c['confidence_threshold']}
    dump(out/'evaluation.json',report)
    training_plots(history,results,c,out,('DEMO ONLY' if demo else 'REAL CANDIDATE')+' / '+c.get('input_mode','time'))
    export(model,out)
    vectors = []
    for i in ids['test'][:10]:
        vectors.append({'record':records[i]['record'], 'preprocessed_input':x[i,:,0].tolist(),
                        'expected_scores':all_probs[i].tolist(),'expected_class_id':int(all_probs[i].argmax()),
                        'model_version':version,'demo_only':demo})
    dump(out/'fixed_test_vectors.json',vectors)
    dump(out/'training_status.json',{'status':status,'model_exported':True,
                                    'fixed_test_vector_count':len(vectors),'esp32_verified':False})
    with (out/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.writer(f); w.writerow(['sample_id','label','group','split','source','release_method'])
        for name,ix in ids.items():
            for i in ix:
                a=records[i]; w.writerow([a['record']['sample_id'],a['label'],a['group'],name,a['source'],a['release_method']])
    card = f"# 两分类模型交付\n\n状态：{status}\n\n版本：{version}\n\n"
    card += "原始输入为16kHz、1024点、单通道，减整窗均值后除2048，float32。\n\n"
    card += f"本模型输入约定：{json.dumps(input_contract(c),ensure_ascii=False)}。时域和频谱模型不能互换输入。\n\n"
    head='保留频率位置并Flatten' if c.get('input_mode')=='spectrum' else '全局平均池化'
    card += f"模型：Conv1D(8,9,2) → ReLU → Conv1D(16,5,2) → ReLU → Conv1D(32,3,2) → ReLU → {head} → 两类softmax。SAME补零，左侧取总补零数的一半向下取整。\n\n"
    card += f"类别顺序：{c['classes']}；显示名称：{c['class_display_names']}。内部编号0/1，显示名称由映射确定，不能套用原占位四分类。\n\n"
    card += "model.json供电脑复现；model_weights.h提供独立C++参考推理及权重。尚未接入固件B或验证ESP32时延/一致性。\n\n"
    card += f"先质量检查，再调用本模型头文件的wood_model_{c.get('input_mode','time')}::preprocess和infer；具体缓冲大小见deployment_budget.json。不要在任务栈上分配大数组。\n\n"
    if c.get('input_mode')=='spectrum':
        card += "频谱头文件用直接DFT提供可读参考，不是高性能FFT。ESP32接入应使用经对照验证的FFT实现替换，Hann、幅值缩放、频点顺序必须一致；频谱变换时延尚未测量。\n\n"
    card += "阈值尚未标定，softmax分数不代表实测正确率。fixed_test_vectors.json用于电脑/固件对照。\n\n"
    card += "若状态DEMO_ONLY，禁止用于实际木棍识别和声称准确率。真实模型仍需李/戴核对数据与采样条件。\n"
    (out/'model_card.md').write_text(card,encoding='utf-8')
    print(json.dumps({'status':status,'test':results['test']['cnn'],'output':str(out)},indent=2))


def demo_data(c,out):
    rng = np.random.default_rng(c['seed'])
    data,meta = [],[]
    t = (np.arange(1024)-256)/16000
    post = np.maximum(t,0)
    for group in range(12):
        perturb = rng.uniform(.96,1.04)
        for label,stick in enumerate(['DEMO_SOLID','DEMO_HOLLOW']):
            for hit in range(1,13):
                position = (hit-1)%3+1
                frequency = (650+label*700)*perturb*rng.uniform(.98,1.02)
                signal = rng.uniform(650,1000)*np.exp(-(45+label*15)*post)*np.sin(2*np.pi*frequency*post)
                signal += 150*np.exp(-60*post)*np.sin(2*np.pi*(frequency*1.7)*post)
                signal[t<0]=0
                raw = np.rint(2048+signal+rng.normal(0,10,1024)).astype(int).tolist()
                sid = build_sample_id(stick,1,group,position,hit)
                data.append({'protocol_version':'WOOD_IMPACT_V1','sample_id':sid,'stick_id':stick,
                             'experiment_batch':1,'reclamp_batch':group,'impact_point_id':position,
                             'strike_id':hit,'sample_rate_hz':16000,'sample_count':1024,'anomaly_flags':[],'raw':raw})
                meta.append([sid,c['classes'][label],'synthetic','manual','false'])
    folder = out/'demo_data'; folder.mkdir()
    path = folder/'demo.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in data)+'\n',encoding='utf-8')
    manifest = folder/'manifest.csv'
    with manifest.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_id','label','source','release_method','electrical_verified']); w.writerows(meta)
    return path,manifest


def predict(model_path,record_path):
    m = json.loads(Path(model_path).read_text(encoding='utf-8'))
    if m.get('format_version') != 'WOOD_CNN_NUMPY_V1':
        raise ValueError('unsupported model format')
    c = config_from_model(m)
    r = json.loads(Path(record_path).read_text(encoding='utf-8-sig'))
    result = {'interface_version':'WOOD_MODEL_OUTPUT_V0_2','sample_id':r.get('sample_id') if isinstance(r,dict) else None,
              'input_mode':c.get('input_mode','time'),'input_interface_version':input_contract(c)['version'],
              'model_version':m['model_version'],'status':'INVALID_SIGNAL','class_id':None,
              'scores':None,'confidence':None,'class_name':None,'demo_only':m['demo_only']}
    try:
        x=preprocess(r,c)
    except (ValueError,TypeError,KeyError) as e:
        result['reason']=str(e)
        return result
    expected_contract=m.get('input_contract')
    if expected_contract is not None and expected_contract!=input_contract(c):
        raise ValueError('model input contract/config mismatch')
    pr=CNN(c,m['weights']).forward(model_input(x[None,:,:],c))[0]
    top=float(pr.max()); cls=int(pr.argmax())
    result.update(scores=pr.tolist(),confidence=top,reason=None)
    if np.count_nonzero(pr==pr.max()) > 1:
        result.update(status='LOW_CONFIDENCE',reason='TIED_TOP_SCORES')
    elif c['confidence_threshold'] is not None and top<c['confidence_threshold']:
        result.update(status='LOW_CONFIDENCE',reason='BELOW_CONFIGURED_THRESHOLD')
    else:
        result.update(status='DEMO_ONLY' if m['demo_only'] else 'OK',class_id=cls,class_name=c['classes'][cls])
    return result


def config_from_model(m):
    # Reuse full config validation without filesystem side effects.
    c=m['config']
    if any(c.get(k)!=v for k,v in {'interface_version':'TIME_V0_1','sample_rate_hz':16000,'sample_count':1024,
                                  'channels':1,'scale_divisor':2048.0,'trigger_index':256}.items()):
        raise ValueError('exported preprocessing contract mismatch')
    if len(c['classes'])!=2 or len(set(c['classes']))!=2:
        raise ValueError('invalid class order')
    if c.get('input_mode','time') not in ['time','spectrum']:
        raise ValueError('invalid model input mode')
    threshold=c.get('confidence_threshold')
    if threshold is not None and (not np.isfinite(threshold) or not 0<=threshold<=1):
        raise ValueError('invalid model confidence threshold')
    return c


def train_candidates(records,c,out,mode='both',demo=False):
    modes=['time','spectrum'] if mode=='both' else [mode]
    summary={'status':'PENDING','selection':'UNDECIDED - use validation and deployment constraints; do not select using test results',
             'same_records':True,'same_group_split_seed':c['seed'],'candidates':{}}
    for choice in modes:
        child=out/choice; child.mkdir()
        candidate_config={**c,'input_mode':choice}
        dump(child/'config_snapshot.json',candidate_config)
        train(records,candidate_config,child,demo)
        status=json.loads((child/'training_status.json').read_text(encoding='utf-8'))
        entry={'input_contract':input_contract(candidate_config),**status}
        if (child/'evaluation.json').exists():
            report=json.loads((child/'evaluation.json').read_text(encoding='utf-8'))
            entry.update(model_version=report['model_version'],best_epoch=report['best_epoch'],
                         validation=report['results']['validation']['cnn'],test=report['results']['test']['cnn'])
        summary['candidates'][choice]=entry
    states=[v['status'] for v in summary['candidates'].values()]
    if all(s!='PENDING' for s in states):
        summary['status']='DEMO_ONLY' if demo else 'REAL_CANDIDATES_REVIEW_REQUIRED'
    dump(out/'candidate_comparison.json',summary)
    dump(out/'training_status.json',{'status':summary['status'],'candidate_modes':modes,
                                   'selection':'UNDECIDED','esp32_verified':False})


def main():
    p=argparse.ArgumentParser(description='Luo 09/28-10/09 offline pipeline')
    p.add_argument('command',choices=['prepare','analyze','train','demo','predict'])
    p.add_argument('--config',default=str(ROOT/'config.json'))
    p.add_argument('--data',nargs='+',default=[str(ROOT/'data'/'real')])
    p.add_argument('--manifest')
    p.add_argument('--out',default=str(ROOT/'runs'/'latest'))
    p.add_argument('--model'); p.add_argument('--record')
    p.add_argument('--input-mode',choices=['time','spectrum','both'],default='both',
                   help='train/demo retain two independent CNN candidates by default')
    a=p.parse_args()
    try:
        if a.command=='predict':
            if not a.model or not a.record:
                p.error('predict needs --model and --record')
            print(json.dumps(predict(a.model,a.record),ensure_ascii=False,indent=2))
            return
        c=config(a.config)
        out=new_output(a.out)
        if a.command=='demo':
            data,manifest=demo_data(c,out)
            records=prepare([data],c,manifest,out)
            train_candidates(records,c,out,a.input_mode,True)
        else:
            records=prepare(a.data,c,a.manifest,out)
            if a.command=='train':
                train_candidates(records,c,out,a.input_mode)
            print(json.dumps({'accepted':len(records),'output':str(out)},indent=2))
    except (ValueError,RuntimeError,OSError) as e:
        print('ERROR: '+str(e),file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':
    main()
