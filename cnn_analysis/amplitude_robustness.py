"""Frozen-weight normalization/robustness experiment; never changes firmware.

Run from repository root with --run and a fresh --out directory.
All normalization targets come from training hits, pooled over endpoints.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .network import CNN
from .triplet import inputs, probabilities, read_data


def amplitude(x, method):
    if method == 'rms':
        return np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1))
    return np.max(np.abs(x), axis=1)


def normalize(x, method, targets):
    if method == 'raw':
        return x.astype(np.float32)
    scale = targets[method] / np.maximum(amplitude(x, method), 1e-12)
    return (x * scale[:, None, :]).astype(np.float32)


def scenarios(x, raw, c):
    for gain in [.25, .5, .75, 1., 1.25, 1.5, 2.]:
        yield 'uniform', str(gain), x * gain, np.ones(len(x), bool)
    gains = list(itertools.product([.5, 1., 1.5], repeat=3))
    for gain in gains:
        yield 'independent', '/'.join(map(str, gain)), x * gain, np.ones(len(x), bool)
    mean = raw.mean(axis=1, keepdims=True)
    noise = np.random.default_rng(1806).normal(size=raw.shape)
    # Keep noise identical across gains and models for paired comparisons.
    for sd in [0, 10, 30, 80]:
        for gain in gains:
            captured = np.clip(np.rint(mean + (raw - mean) * gain + noise * sd), 0, 4095)
            accepted = ((captured.min(axis=(1, 2)) > c['clip_low']) &
                        (captured.max(axis=(1, 2)) < c['clip_high']) &
                        (np.ptp(captured, axis=1).min(axis=1) >= c['min_peak_to_peak']))
            centered = (captured.astype(np.float32) - captured.mean(axis=1, keepdims=True).astype(np.float32)) / 2048
            yield 'adc_noise_' + str(sd), '/'.join(map(str, gain)), centered, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exclude-sample', action='append', default=[], help='Exclude whole triplets containing this exact sample ID; preserve source data.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    c = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    audit = json.loads((args.run / 'data_audit.json').read_text(encoding='utf-8'))
    for f in audit['files']:
        assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() == f['sha256']
    groups, time, y, _ = read_data([f['path'] for f in audit['files']], c)
    splits = json.loads((args.run / 'split_groups.json').read_text(encoding='utf-8'))
    lookup = {g['id']: i for i, g in enumerate(groups)}
    original_ids = {name: np.array([lookup[s] for s in values]) for name, values in splits.items()}
    known_samples = {r['sample_id'] for g in groups for r in g['records']}
    if set(args.exclude_sample) - known_samples:
        raise ValueError('Unknown excluded sample ID')
    excluded_groups = {g['id'] for g in groups if any(r['sample_id'] in args.exclude_sample for r in g['records'])}
    excluded_by_split = {s: [v for v in values if v in excluded_groups] for s, values in splits.items()}
    splits = {s: [v for v in values if v not in excluded_groups] for s, values in splits.items()}
    ids = {name: np.array([lookup[s] for s in values]) for name, values in splits.items()}
    if any(not len(ix) or len(set(y[ix])) != 2 for ix in ids.values()):
        raise ValueError('Exclusion leaves an empty partition or missing class')
    assert all(not set(ids[a]) & set(ids[b]) for a, b in itertools.combinations(ids, 2))
    targets = {}
    for method in ['rms', 'peak']:
        a = amplitude(time[ids['train']], method)
        endpoint = np.median(a[:, [0, 2]])
        targets[method] = np.array([endpoint, np.median(a[:, 1]), endpoint])
        n = normalize(time, method, targets)
        assert np.allclose(amplitude(n, method), targets[method], atol=1e-7)
        assert np.allclose(normalize(time[:, :, ::-1], method, targets), n[:, :, ::-1], atol=1e-7)
        assert np.allclose(normalize(time * [.5, 1., 1.5], method, targets), n, atol=1e-6)
    models = {}
    fingerprints = {}
    manifest = json.loads(Path('firmware_b_demo/include/models/manifest.json').read_text(encoding='utf-8'))
    for mode in ['time', 'spectrum']:
        path = args.run / mode / 'model.json'
        model = json.loads(path.read_text(encoding='utf-8'))
        assert model['model_version'] == manifest['models'][mode]['version']
        models[mode] = (CNN({**model['config'], 'channels': 3}, model['weights']), model['config'])
        fingerprints[mode] = {'version': model['model_version'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    records, predictions, summary = [], [], {}
    for split in ['validation', 'test']:
        ix = ids[split]
        x, labels = time[ix], y[ix]
        # Generate the original perturbations before filtering, so every retained
        # sample receives exactly the same noise as in the unfiltered audit.
        source_ix = original_ids[split]
        keep = np.isin(source_ix, ix)
        raw = np.stack([np.stack([r['raw'] for r in groups[i]['records']], axis=1) for i in source_ix])
        cases = [(suite, condition, values[keep], accepted[keep])
                 for suite, condition, values, accepted in scenarios(time[source_ix], raw, c)]
        all_inputs = np.concatenate([case[2] for case in cases])
        summary[split] = {}
        for mode, (net, mc) in models.items():
            for method in ['raw', 'rms', 'peak']:
                name = mode + '/' + method
                pp = probabilities(net, inputs(normalize(all_inputs, method, targets), mc)).reshape(len(cases), len(x), 2)
                baseline = probabilities(net, inputs(normalize(x, method, targets), mc))
                rev = probabilities(net, inputs(normalize(x[:, :, ::-1], method, targets), mc))
                assert np.max(np.abs(rev - baseline)) < 1e-6
                rows = []
                for case, p in zip(cases, pp):
                    suite, condition, _, accepted = case
                    correct = p.argmax(1) == labels
                    row = {'split': split, 'model': mode, 'normalization': method, 'suite': suite,
                           'condition': condition, 'count': len(x), 'correct': int(correct.sum()),
                           'flips': int((p.argmax(1) != baseline.argmax(1)).sum()),
                           'mean_probability_drift': float(np.abs(p[:, 1] - baseline[:, 1]).mean()),
                           'max_probability_drift': float(np.abs(p[:, 1] - baseline[:, 1]).max()),
                           'accepted': int(accepted.sum()), 'accepted_correct': int(correct[accepted].sum()),
                           'wood01_correct': int(correct[labels == 0].sum()),
                           'wood02_correct': int(correct[labels == 1].sum())}
                    records.append(row); rows.append(row)
                    predictions.append({**{k: row[k] for k in ['split', 'model', 'normalization', 'suite', 'condition']},
                                        'p_wood02': p[:, 1].tolist(), 'accepted': accepted.tolist()})
                sub = {'baseline_correct': int((baseline.argmax(1) == labels).sum()), 'count': len(x), 'suites': {}}
                for suite in dict.fromkeys(r['suite'] for r in rows):
                    rr = [r for r in rows if r['suite'] == suite]
                    sub['suites'][suite] = {'conditions': len(rr), 'worst_correct': min(r['correct'] for r in rr),
                        'correct_sum': sum(r['correct'] for r in rr), 'count_sum': sum(r['count'] for r in rr),
                        'flips_sum': sum(r['flips'] for r in rr), 'max_probability_drift': max(r['max_probability_drift'] for r in rr),
                        'accepted_sum': sum(r['accepted'] for r in rr),
                        'accepted_correct_sum': sum(r['accepted_correct'] for r in rr)}
                summary[split][name] = sub
                print(split, name, sub['baseline_correct'], sub['suites']['independent'], flush=True)
        if split == 'validation':
            # Predeclared selection: best worst-case ideal independent-gain accuracy,
            # then baseline accuracy, then accepted noisy accuracy; no held-out tuning.
            def key(name):
                a = summary['validation'][name]
                s = a['suites']['adc_noise_30']
                return (a['suites']['independent']['worst_correct'], a['baseline_correct'],
                        s['accepted_correct_sum'] / max(s['accepted_sum'], 1))
            ranking = sorted(summary['validation'], key=key, reverse=True)
            lock = {'ranking': ranking, 'selected': ranking[0], 'selection_rule': 'worst ideal independent-gain accuracy; baseline accuracy; accepted ADC noise-30 accuracy',
                    'scores': {name: key(name) for name in ranking}, 'note': 'Selection written before held-out test evaluation; parent models already have historical test exposure.'}
            (args.out / 'selection_locked.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    result = {'models': fingerprints, 'targets': {k: v.tolist() for k, v in targets.items()},
              'excluded_sample_ids': args.exclude_sample, 'excluded_triplets_by_split': excluded_by_split,
              'normalization': 'Per-hit whole-window amplitude scaled to train median; endpoints pooled to preserve reversal invariance. Before Hann/FFT for spectrum; weights frozen.',
              'split_ids': splits, 'true_labels': {s: y[ix].tolist() for s, ix in ids.items()},
              'summary': summary, 'selection': lock,
              'limitations': ['Synthetic amplitude, fixed additive noise, rounding and clipping; not new impacts at measured forces.',
                'Original matching-H triplets after explicit exclusions; repeated perturbations are dependent observations.',
                'Existing normal/abnormal quality model not included. Hard acceptance checks only clipping/weak signal.',
                'Training and test parent model history already viewed; retrospective comparison, not fresh independent validation.',
                'No retraining, firmware change, or upload. Zero/weak signals should be rejected before normalization.']}
    (args.out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
    with (args.out / 'conditions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_report(args.out, result, records)


def write_report(out, result, records):
    names = {'time/raw': '时域 · 不归一化', 'time/rms': '时域 · 均方根归一化', 'time/peak': '时域 · 峰值归一化',
             'spectrum/raw': '频域 · 不归一化', 'spectrum/rms': '频域 · 均方根归一化', 'spectrum/peak': '频域 · 峰值归一化'}
    lines = ['# 振幅归一化与鲁棒性对比', '', '现有权重保持不变；两套模型均在时域去均值后归一化，频域模型随后执行原有 Hann 窗和 FFT。每次敲击单独缩放到训练集典型振幅，两端共用目标值。均方根代表整体信号能量尺度；峰值代表最大绝对幅值。', '',
        f"验证集{len(result['split_ids']['validation'])}组，测试集{len(result['split_ids']['test'])}组三点输入，每组3次敲击。重复扰动不是新增独立样本。整体缩放0.25–2倍；各位置独立取0.5、1、1.5倍，穷举27种组合。归一化目标只从训练集计算。", '',
        '验证集按预先写定的规则排序，在测试计算前保存 selection_locked.json。由于历史上已经查看过母模型的测试集表现，本次仍属回顾性诊断。验证集选定：' + names[result['selection']['selected']] + '。', '']
    html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>振幅归一化鲁棒性测试</title><style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1140px;margin:36px auto;background:#f5f8fa;color:#183047}section{background:white;border-radius:16px;padding:24px;margin:20px 0}p{line-height:1.8}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px;border-bottom:1px solid #dae3ea;text-align:left}h1{font-size:30px}</style><h1>先归一化，能否让判断更稳定？</h1><p>现有时域与频域模型 · 权重冻结 · 训练集确定目标振幅 · 每个评估分区20组三点输入</p>']
    html[0] = html[0].replace('每个评估分区20组三点输入', f"验证集{len(result['split_ids']['validation'])}组 · 测试集{len(result['split_ids']['test'])}组")
    if result.get('excluded_sample_ids'):
        note = '本版排除已标记异常记录 ' + '、'.join(result['excluded_sample_ids']) + ' 所属的整组三点输入。保留原始数据、历史报告及其余样本的原模拟噪声。未重新训练；这是按已知异常标记进行的回顾性过滤。'
        lines += [note, '']
        html.append('<section><h2>本版数据排除</h2><p>' + note + '</p></section>')
    for split, title in [('validation', '验证集'), ('test', '留出测试集')]:
        lines += ['## ' + title, '', '| 方案 | 原始输入正确 | 整体缩放最差正确 | 各位置独立缩放最差正确 | 独立缩放预测翻转 |', '|---|---:|---:|---:|---:|']
        html.append(f'<section><h2>{title}：理想振幅变化</h2><table><tr><th>方案</th><th>原始输入</th><th>整体缩放最差</th><th>各位置独立缩放最差</th><th>预测翻转</th></tr>')
        for name, a in result['summary'][split].items():
            u, i = a['suites']['uniform'], a['suites']['independent']
            vals = [names[name], f'{a["baseline_correct"]}/{a["count"]}', f'{u["worst_correct"]}/{a["count"]}', f'{i["worst_correct"]}/{a["count"]}', f'{i["flips_sum"]}/{i["count_sum"]}']
            lines.append('| ' + ' | '.join(vals) + ' |')
            html.append('<tr>' + ''.join(f'<td>{v}</td>' for v in vals) + '</tr>')
        html.append('</table><p>“翻转”相对同一方案未经扰动的输出计算。预测稳定不等于预测正确。</p></section>')
        lines.append('')
    html.append('<section><h2>留出测试：整体振幅缩放下的正确率</h2><svg viewBox="0 0 1060 380" role="img" aria-label="振幅缩放正确率曲线">')
    colors = ['#bf4e30', '#007f82', '#68a62c', '#9c52b0', '#2867c7', '#bd7f00']
    for tick in [0, .25, .5, .75, 1]:
        yy = 290 - tick * 230
        html.append(f'<line x1="70" y1="{yy}" x2="710" y2="{yy}" stroke="#dde4ea"/><text x="15" y="{yy+5}" font-size="15">{tick:.0%}</text>')
    for idx, (name, color) in enumerate(zip(names, colors)):
        mode, norm = name.split('/')
        rr = [r for r in records if r['split']=='test' and r['model']==mode and r['normalization']==norm and r['suite']=='uniform']
        pts = ' '.join(f'{70+(float(r["condition"])-.25)/1.75*640},{290-r["correct"]/r["count"]*230}' for r in rr)
        dash = ' stroke-dasharray="8 5"' if mode == 'spectrum' else ''
        html.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"{dash}/><text x="735" y="{70+idx*34}" fill="{color}" font-size="16">{names[name]}</text>')
    for gain in [.25,.5,.75,1,1.25,1.5,2]:
        html.append(f'<text x="{60+(gain-.25)/1.75*640}" y="320" font-size="16">{gain}×</text>')
    html.append('<text x="250" y="360" font-size="16">整体振幅倍率（相同曲线会重叠）</text></svg></section>')
    lines += ['## 采集限制与固定噪声', '', '模拟原始采集值围绕均值缩放，添加固定标准差噪声，四舍五入并限制到0–4095；先检测削顶与弱信号。以下只统计通过这两项检查的组合；未包含完整质量模型。所有方案使用完全相同的扰动。', '']
    html.append('<section><h2>采集限制与固定噪声：通过基本质量检查后的正确率</h2><p>未通过削顶/弱信号检查的输入应重测，不能靠归一化修复。各噪声级使用27种位置振幅组合；下表中的次数来自20组原始输入的重复扰动。</p><table><tr><th>方案</th><th>无新增噪声</th><th>噪声10计数</th><th>噪声30计数</th><th>噪声80计数</th></tr>')
    html[-1] = html[-1].replace('来自20组', f"来自{len(result['split_ids']['test'])}组")
    lines += ['| 方案 | 无新增噪声 | 噪声10计数 | 噪声30计数 | 噪声80计数 |', '|---|---:|---:|---:|---:|']
    for name, a in result['summary']['test'].items():
        vals = [names[name]]
        for sd in [0, 10, 30, 80]:
            s = a['suites']['adc_noise_' + str(sd)]
            vals.append(f'{s["accepted_correct_sum"]}/{s["accepted_sum"]}')
        lines.append('| ' + ' | '.join(vals) + ' |')
        html.append('<tr>' + ''.join(f'<td>{v}</td>' for v in vals) + '</tr>')
    note = '理想的正比例缩放会被逐次归一化在数学上抵消，因此“零翻转”不能证明真实敲击力度鲁棒。真实力度还可能改变接触时长、频谱、噪声占比并导致削顶。这里的推荐仅适用于这两根木头和合成扰动；需要补采轻、中、重敲击验证。弱信号应先拒收，避免把噪声放大；削顶信息无法恢复。未修改或刷写固件。'
    html.append(f'</table></section><section><h2>解释边界与下一步</h2><p>{note}</p></section></html>')
    lines += ['', '## 解释边界', '', note, '', '详细条件见 conditions.csv；逐样本概率及有效标记见 predictions.json；参数、模型指纹与汇总见 results.json。']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'report.html').write_text(''.join(html), encoding='utf-8')


if __name__ == '__main__':
    main()
