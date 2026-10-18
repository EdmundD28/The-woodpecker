"""Train and evaluate three-position CNNs for two known wooden specimens."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .core import config, dump, preprocess, validate_config
from .export import export
from .network import CNN, signal_features, fit_centroid, centroid_predict
from .reporting import metrics, plotter

FORMAT = 'WOOD_TRIPLET_CNN_V1'
POINTS = [1, 2, 3]


def inputs(time, c):
    if time.ndim != 3 or time.shape[1:] != (c['sample_count'], 3):
        raise ValueError('three inputs required: one endpoint, middle, other endpoint')
    if c['input_mode'] == 'time':
        return time.astype(np.float32, copy=False)
    if 'spectrum_preparation' in c:
        from .spectrum_preparation import rms_log
        return rms_log(time,c,c['spectrum_preparation'])
    windowed = time.astype(np.float64)*np.hanning(c['sample_count'])[None, :, None]
    return (np.abs(np.fft.rfft(windowed, axis=1))/c['sample_count']).astype(np.float32)


def probabilities(net, x, batch_size=32):
    """Exactly invariant to exchanging the two endpoints, with middle fixed."""
    rows = []
    for start in range(0, len(x), batch_size):
        b = x[start:start+batch_size]
        rows.append((net.forward(b)+net.forward(b[:, :, ::-1]))*.5)
    return np.concatenate(rows)


def scores(y, p):
    return {**metrics(y, p.argmax(1)),
            'cross_entropy': float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())}


def read_data(paths, c):
    grouped, seen_ids, seen_raw, files = {}, set(), {}, []
    for path in paths:
        path = Path(path)
        content = path.read_bytes()
        count = 0
        for line in content.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            preprocess(r, c)  # Refuse rather than silently drop an invalid hit.
            if r['stick_id'] not in c['classes']:
                raise ValueError('unexpected specimen: '+r['stick_id'])
            if r['sample_id'] in seen_ids:
                raise ValueError('duplicate sample_id: '+r['sample_id'])
            digest = hashlib.sha256(np.asarray(r['raw'], dtype='<u2').tobytes()).hexdigest()
            if digest in seen_raw:
                raise ValueError('duplicate raw waveforms: '+seen_raw[digest]+' / '+r['sample_id'])
            seen_ids.add(r['sample_id']); seen_raw[digest] = r['sample_id']
            key = (r['stick_id'], r['experiment_batch'], r['reclamp_batch'], r['strike_id'])
            points = grouped.setdefault(key, {})
            if r['impact_point_id'] in points:
                raise ValueError('duplicate point in triplet')
            points[r['impact_point_id']] = r
            count += 1
        files.append({'path': str(path.resolve()), 'records': count,
                      'sha256': hashlib.sha256(content).hexdigest()})
    groups = []
    for key, points in sorted(grouped.items()):
        if set(points) != set(POINTS):
            raise ValueError('incomplete three-point group: '+str(key))
        groups.append({'id': f'{key[0]}-E{key[1]:02}-R{key[2]:02}-H{key[3]:03}',
                       'label': c['classes'].index(key[0]), 'records': [points[p] for p in POINTS]})
    if not groups:
        raise ValueError('no complete triplets')
    time = np.stack([np.concatenate([preprocess(r, c) for r in g['records']], axis=1) for g in groups])
    y = np.array([g['label'] for g in groups])
    audit = {'files': files, 'records': len(seen_ids), 'unique_raw_waveforms': len(seen_raw),
             'complete_triplets': len(groups), 'source': 'real, confirmed by user',
             'electrical_verified': False, 'point_order': POINTS,
             'pairing': 'Matching H across separately collected P01/P02/P03 blocks; not contemporaneous display trials'}
    return groups, time, y, audit


def partition(groups, c):
    rng = np.random.default_rng(c['seed'])
    parts = {'train': [], 'validation': [], 'test': []}
    for label in range(2):
        pool = np.array([i for i, g in enumerate(groups) if g['label'] == label])
        if len(pool) < 5:
            raise ValueError('need at least five complete triplets per specimen')
        pool = rng.permutation(pool)
        n = max(1, len(pool)//5)
        parts['test'].extend(pool[:n]); parts['validation'].extend(pool[n:2*n])
        parts['train'].extend(pool[2*n:])
    return {name: np.asarray(ix, dtype=int) for name, ix in parts.items()}


def combinations(groups, ids, random_count=None, seed=42):
    """Each row has exactly one P01, one P02, one P03 from one specimen."""
    rng = np.random.default_rng(seed)
    result = []
    for label in range(2):
        pool = [int(i) for i in ids if groups[i]['label'] == label]
        if random_count is None:
            result.extend(itertools.product(pool, repeat=3))
        else:
            # Include every source hit once before drawing additional combinations.
            rows = [(i, i, i) for i in pool]
            target = max(random_count, len(pool))
            seen = set(rows)
            while len(rows) < min(target, len(pool)**3):
                row = tuple(int(v) for v in rng.choice(pool, size=3))
                if row not in seen:
                    rows.append(row); seen.add(row)
            result.extend(rows)
    return np.asarray(result, dtype=int)


def assemble(time, groups, rows):
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError('three positional source indices required')
    y = np.array([groups[i]['label'] for i in rows[:, 0]])
    for p in range(1, 3):
        if any(groups[i]['label'] != label for i, label in zip(rows[:, p], y)):
            raise ValueError('cannot mix specimens in a triplet')
    return np.stack([time[rows[:, p], :, p] for p in range(3)], axis=2), y


def contract(c):
    length = c['sample_count'] if c['input_mode'] == 'time' else c['sample_count']//2+1
    result = {'version': FORMAT, 'mode': c['input_mode'], 'shape': [length, 3],
            'sample_rate_hz': c['sample_rate_hz'], 'raw_shape': [3, c['sample_count']],
            'positions': 'endpoint / middle / other endpoint; P01/P02/P03 or P03/P02/P01',
            'direction_fusion': 'arithmetic mean of forward and endpoint-reversed softmax scores',
            'preprocessing': 'independent whole-window mean removal and /2048 for each hit',
            'spectrum': 'symmetric Hann, abs(rfft)/sample_count, no phase or peak normalization' if c['input_mode']=='spectrum' else None}
    if 'spectrum_preparation' in c:
        result.update(version='WOOD_TRIPLET_RMS_LOG_V1',spectrum='per-hit RMS normalization, symmetric Hann, abs(rfft)/sample_count, fitted log1p compression',
                      spectrum_preparation=c['spectrum_preparation'])
    return result


def export_triplet(model, out):
    export(model, out, input_channels=3)
    mode = model['config']['input_mode']
    length = model['input_contract']['shape'][0]
    header_path = out/'model_weights.h'
    header = header_path.read_text(encoding='utf-8')
    if 'spectrum_preparation' in model['config']:
        from .export_rms_log import patch_header
        header=patch_header(header,model)
    wrapper = f'''// Use this entry point for direction-independent three-hit classification.
// reverse_input requires {length*3} floats, separate from input and scratch.
inline void infer_either_direction(const float* input,float* scores,float* scratch,float* reverse_input) {{
    for(int i=0;i<{length};++i) for(int p=0;p<3;++p) reverse_input[i*3+p]=input[i*3+2-p];
    float reversed_scores[2];
    infer(input,scores,scratch); infer(reverse_input,reversed_scores,scratch);
    for(int p=0;p<2;++p) scores[p]=(scores[p]+reversed_scores[p])*0.5f;
}}
'''
    end = header.rfind('} // namespace')
    header_path.write_text(header[:end]+wrapper+header[end:], encoding='utf-8')
    budget = json.loads((out/'deployment_budget.json').read_text(encoding='utf-8'))
    budget.update(direction_invariant_conv_dense_macs=budget['conv_dense_macs']*2,
                  reverse_input_bytes=length*3*4, namespace='wood_triplet_'+mode,
                  required_inference='preprocess_three then infer_either_direction')
    if 'spectrum_preparation' in model['config']:
        budget.update(spectrum_preparation=model['config']['spectrum_preparation'],
                      fitted_preparation_constant_bytes=72,additional_runtime_scratch_bytes=0)
    dump(out/'deployment_budget.json', budget)


def evaluate_combinations(net, time, groups, rows, c, baselines=None):
    labels, probs, baseline_predictions = [], [], {k: [] for k in baselines or {}}
    for start in range(0, len(rows), 32):
        x, y = assemble(time, groups, rows[start:start+32])
        labels.append(y); probs.append(probabilities(net, inputs(x, c)))
        for name, model in (baselines or {}).items():
            baseline_predictions[name].append(centroid_predict(features(x, name), model))
    y, p = np.concatenate(labels), np.concatenate(probs)
    report = scores(y, p)
    report['per_specimen'] = {c['classes'][label]: scores(y[y==label], p[y==label]) for label in range(2)}
    report['baselines'] = {name: metrics(y, np.concatenate(pred)) for name, pred in baseline_predictions.items()}
    return report, p


def features(time, mode):
    if mode == 'amplitude_only':
        peak = np.ptp(time, axis=1)
        rms = np.sqrt(np.mean(time*time, axis=1))
        return np.stack([(peak[:,0]+peak[:,2])*.5, peak[:,1],
                         (rms[:,0]+rms[:,2])*.5, rms[:,1]], axis=1)
    # Average the endpoint features to preserve the same reversal invariance.
    f = [signal_features(time[:, :, p:p+1], True) for p in range(3)]
    return np.concatenate([(f[0]+f[2])*.5, f[1]], axis=1)


def train_run(paths, out, epochs=120):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    c = config()
    c.update(classes=['WOOD01', 'WOOD02'], class_display_names=['木头1', '木头2'],
             stick_labels={'WOOD01': 'WOOD01', 'WOOD02': 'WOOD02'}, epochs=epochs, patience=20)
    groups, time, y, audit = read_data(paths, c)
    parts = partition(groups, c)
    dump(out/'data_audit.json', audit)
    dump(out/'config.json', c)
    dump(out/'split_groups.json', {name: [groups[i]['id'] for i in ix] for name, ix in parts.items()})
    with (out/'split_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f); writer.writerow(['split', 'triplet', 'label', 'point', 'sample_id'])
        for name, ids in parts.items():
            for i in ids:
                for r in groups[i]['records']:
                    writer.writerow([name, groups[i]['id'], r['stick_id'], r['impact_point_id'], r['sample_id']])
    train_rows = combinations(groups, parts['train'], random_count=120, seed=c['seed'])
    tx, ty = assemble(time, groups, train_rows)
    dump(out/'training_combinations.json', [[groups[i]['records'][p]['sample_id'] for p, i in enumerate(row)] for row in train_rows])
    baselines = {name: fit_centroid(features(tx, name), ty) for name in ['amplitude_only', 'shape_without_amplitude']}
    dump(out/'baseline_models.json', baselines)
    validation_rows = combinations(groups, parts['validation'])
    candidates = {}
    for mode in ['time', 'spectrum']:
        child = out/mode; child.mkdir()
        cc = {**c, 'input_mode': mode}
        net = CNN({**cc, 'channels': 3})
        x = inputs(tx, cc)
        vx = inputs(time[parts['validation']], cc); vy = y[parts['validation']]
        history = net.train(np.concatenate([x, x[:, :, ::-1]]), np.tile(ty, 2),
                            np.concatenate([vx, vx[:, :, ::-1]]), np.tile(vy, 2), cc)
        validation, _ = evaluate_combinations(net, time, groups, validation_rows, cc)
        weights = {name: value.tolist() for name, value in net.p.items()}
        model = {'format_version': FORMAT, 'config': cc, 'network_channels': 3,
                 'point_order': POINTS, 'input_contract': contract(cc), 'weights': weights,
                 'model_version': 'REAL-TRIPLET-'+hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:12],
                 'status': 'REAL_CANDIDATE_ESP32_NOT_VERIFIED', 'source_audit': audit['files']}
        dump(child/'model.json', model); dump(child/'history.json', history)
        dump(child/'validation.json', validation)
        export_triplet(model, child)
        candidates[mode] = {'net': net, 'config': cc, 'validation': validation, 'model': model}
        print(json.dumps({'mode': mode, 'epochs': len(history), 'validation': validation['accuracy']}, ensure_ascii=False), flush=True)
    # Lock selection using validation only, before computing any test predictions.
    chosen = min(candidates, key=lambda mode: (-candidates[mode]['validation']['balanced_accuracy'],
                                              candidates[mode]['validation']['cross_entropy']))
    dump(out/'selection_locked.json', {'chosen_mode': chosen, 'basis': 'validation balanced accuracy, then validation cross-entropy',
                                       'test_predictions_computed': False})
    test_rows = combinations(groups, parts['test'])
    summary = {'chosen_mode': chosen, 'classes': c['classes'], 'point_order': POINTS,
               'direction_invariant': True, 'source_records': audit['records'],
               'source_triplets': len(groups), 'split_raw_hits': {name: len(ix)*3 for name, ix in parts.items()},
               'training_combinations_before_reversal': len(train_rows),
               'training_examples_with_reversal': len(train_rows)*2,
               'validation_combinations': len(validation_rows), 'test_combinations': len(test_rows),
               'test_combinations_are_independent': False, 'candidates': {}}
    for mode, candidate in candidates.items():
        net, cc = candidate['net'], candidate['config']
        test, p = evaluate_combinations(net, time, groups, test_rows, cc, baselines)
        fixed_p = probabilities(net, inputs(time[parts['test']], cc))
        entry = {'validation': candidate['validation'], 'test_combinations': test,
                 'test_disjoint_triplets': scores(y[parts['test']], fixed_p),
                 'parameters': sum(v.size for v in net.p.values())}
        summary['candidates'][mode] = entry
        dump(out/mode/'evaluation.json', entry)
        predictions = []
        for row, prob in zip(test_rows, p):
            actual = groups[row[0]]['label']
            predictions.append({'sample_ids': [groups[i]['records'][pos]['sample_id'] for pos, i in enumerate(row)],
                                'actual': c['classes'][actual], 'predicted': c['classes'][int(prob.argmax())],
                                'scores': prob.tolist(), 'correct': int(prob.argmax()) == actual})
        dump(out/mode/'test_predictions.json', predictions)
        vectors = []
        vector_ids = np.concatenate([parts['test'][y[parts['test']]==label][:3] for label in range(2)])
        for i in vector_ids:
            xx = inputs(time[i:i+1], cc)
            vectors.append({'records': groups[i]['records'], 'preprocessed_input': xx[0].tolist(),
                            'expected_scores': probabilities(net, xx)[0].tolist()})
        dump(out/mode/'fixed_test_vectors.json', vectors)
    dump(out/'summary.json', summary)
    make_report(out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def make_report(out, summary):
    selected = summary['candidates'][summary['chosen_mode']]
    text = '# 10月17日木头编号识别：三点联合训练\n\n'
    text += '每组一条点1、一条点2、一条点3；两端可交换，中点保持在中间。正反向数据参加训练，推理平均正反向分数。\n\n'
    parts = summary['split_raw_hits']
    text += f"原始记录{summary['source_records']}条，共{summary['source_triplets']}组不复用波形的三点组。训练{parts['train']}条、验证{parts['validation']}条、测试{parts['test']}条。\n\n"
    text += f"训练池组合{summary['training_combinations_before_reversal']}组，加反向后{summary['training_examples_with_reversal']}组；仅复用训练池内部波形。\n\n"
    text += f"测试为每根木头各点10条留出记录的全部10×10×10组合，共{summary['test_combinations']}组；它们来自60条留出敲击，不是2000次独立实测。另报告20组互不复用波形的测试。\n\n"
    text += '数据按点分批采集，因此组合模拟展示流程，并非实际连续敲完三个点的展示试验。模型只读取波形，不读取木头编号作为输入。\n\n'
    text += '| 模型 | 参数 | 验证组合准确率 | 留出组合准确率 | 不复用波形的测试 |\n|---|---:|---:|---:|---:|\n'
    for mode, e in summary['candidates'].items():
        fixed = e['test_disjoint_triplets']; correct = round(fixed['accuracy']*fixed['count'])
        name = '时域' if mode == 'time' else '频谱'
        text += f"| {name} | {e['parameters']} | {e['validation']['accuracy']:.2%} | {e['test_combinations']['accuracy']:.2%} | {correct}/{fixed['count']} |\n"
    chosen_name = '时域' if summary['chosen_mode'] == 'time' else '频谱'
    text += f"\n按验证集选定：**{chosen_name}模型**，选定后才计算测试结果，未按测试成绩调参。两套权重均保留；时域在这批留出组合上更好，需新增现场数据再确认选型。\n\n"
    text += '选定模型的测试混淆矩阵（行实际、列预测，顺序木头1、木头2）：\n\n'
    text += '```text\n'+str(selected['test_combinations']['confusion_matrix_rows_actual_columns_predicted'])+'\n```\n\n'
    text += '同一测试组合上的简单对照：\n\n'
    for name, result in selected['test_combinations']['baselines'].items():
        display_name = '仅看振幅' if name == 'amplitude_only' else '去掉振幅后的形状特征'
        text += f"- {display_name}：{result['accuracy']:.2%}\n"
    from collections import Counter
    rows = json.loads((out/summary['chosen_mode']/'test_predictions.json').read_text(encoding='utf-8'))
    bad = [r for r in rows if not r['correct']]
    counts = Counter(s for r in rows for s in r['sample_ids'])
    errors = Counter(s for r in bad for s in r['sample_ids'])
    diagnosis = [{'sample_id': sid, 'wrong_combinations': n, 'total_combinations': counts[sid]}
                 for sid, n in errors.most_common(10)]
    dump(out/'error_analysis.json', {'selected_mode': summary['chosen_mode'], 'wrong_combinations': len(bad),
                                    'most_involved_hits': diagnosis, 'records_removed': False})
    if diagnosis:
        top = diagnosis[0]
        text += f"\n错误组合共{len(bad)}组。其中 `{top['sample_id']}` 出现在{top['wrong_combinations']}个错误组合中；包含该记录的组合共{top['total_combinations']}组。\n"
        text += '这是错误关联定位，不能直接认定该记录无效。未删除记录、未据此重训；原测试结果完整保留。\n'
    text += '\n局限：仅限这两根木头、这次采集条件。交换输入两端的结果一致，不等于已经验证物理翻面、重新装夹或换日期后的波形变化。ESP32部署与现场连续三击仍待实测。\n'
    (out/'report.md').write_text(text, encoding='utf-8')
    plt = plotter()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, mode in zip(axes[:2], ['time', 'spectrum']):
        history = json.loads((out/mode/'history.json').read_text(encoding='utf-8'))
        ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='train')
        ax.plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], label='validation')
        ax.set(title=mode, xlabel='Epoch', ylabel='Cross-entropy'); ax.legend(); ax.grid(alpha=.2)
    cm = np.array(selected['test_combinations']['confusion_matrix_rows_actual_columns_predicted'])
    axes[2].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): axes[2].text(j, i, str(cm[i, j]), ha='center', va='center')
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['WOOD01', 'WOOD02'], yticklabels=['WOOD01', 'WOOD02'],
                xlabel='Predicted', ylabel='Actual', title='Held-out combinations / '+summary['chosen_mode'])
    fig.tight_layout(); fig.savefig(out/'training_results.png', dpi=160); plt.close(fig)


def predict(model_path, records):
    model = json.loads(Path(model_path).read_text(encoding='utf-8'))
    if model.get('format_version') != FORMAT or model.get('network_channels') != 3 or model.get('point_order') != POINTS:
        raise ValueError('not a supported three-point model')
    c = validate_config(model['config'])
    if model.get('input_contract') != contract(c):
        raise ValueError('three-point model input contract mismatch')
    order = [r['impact_point_id'] for r in records]
    if order not in [POINTS, POINTS[::-1]]:
        raise ValueError('need one endpoint, point 2, then the other endpoint: 123 or 321')
    x = np.concatenate([preprocess(r, c) for r in records], axis=1)[None, :, :]
    net = CNN({**c, 'channels': 3}, model['weights'])
    p = probabilities(net, inputs(x, c))[0]
    return {'model_version': model['model_version'], 'predicted': c['classes'][int(p.argmax())],
            'display_name': c['class_display_names'][int(p.argmax())], 'scores': p.tolist(),
            'score_note': 'uncalibrated model scores, not measured probability of correctness',
            'point_order_received': order, 'direction_invariant': True}


def main():
    parser = argparse.ArgumentParser(description='Known-board identification from three ordered positions, reversible endpoints')
    commands = parser.add_subparsers(dest='command', required=True)
    tr = commands.add_parser('train'); tr.add_argument('--data', nargs='+', required=True)
    tr.add_argument('--out', required=True); tr.add_argument('--epochs', type=int, default=120)
    pr = commands.add_parser('predict'); pr.add_argument('--model', required=True)
    pr.add_argument('--records', nargs=3, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train_run(args.data, args.out, args.epochs)
    else:
        records = [json.loads(Path(p).read_text(encoding='utf-8-sig')) for p in args.records]
        print(json.dumps(predict(args.model, records), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
