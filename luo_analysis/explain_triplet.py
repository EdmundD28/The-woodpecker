"""Exact grouped interventional Shapley audit of the deployed triplet CNN.

Enumerates every coalition; this is grouped SHAP, not a fitted surrogate.
Only training triplets provide backgrounds. Model weights are never changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .network import CNN
from .triplet import read_data, probabilities


def shapley(values, features):
    result = np.zeros((*values.shape[:-1], features), dtype=np.float64)
    for j in range(features):
        for mask in range(1 << features):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = 1 / (features * math.comb(features - 1, size))
            result[..., j] += weight * (values[..., mask | (1 << j)] - values[..., mask])
    return result


def audit(net, x, background, mode):
    if mode == 'position':
        labels = ['位置1（端点）', '位置2（中点）', '位置3（端点）']
        mapping = np.broadcast_to(np.arange(3)[None, :], x.shape[1:])
        xx, bb = x, background
    elif mode == 'time':
        # Relative to trigger at sample 256, with all three channels grouped.
        edges = [0, 256, 384, 640, 1024, 1600, 2400]
        labels = ['触发前 −32–0 毫秒', '触发后 0–16 毫秒', '触发后 16–48 毫秒',
                  '触发后 48–96 毫秒', '触发后 96–168 毫秒', '触发后 168–268 毫秒']
        mapping = np.zeros(x.shape[1:], dtype=int)
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            mapping[a:b] = j
        xx, bb = x, background
    else:
        edges = [0, 250, 500, 1000, 2000, 4001]
        labels = ['0–250 Hz', '250–500 Hz', '500–1000 Hz', '1000–2000 Hz', '2000–4000 Hz']
        freq = np.fft.rfftfreq(x.shape[1], 1 / 8000)
        mapping = np.broadcast_to(np.searchsorted(edges[1:], freq, side='right')[:, None], (len(freq), 3))
        # Preserve complex phase; inverse FFT feeds the ORIGINAL time model.
        xx, bb = np.fft.rfft(x, axis=1), np.fft.rfft(background, axis=1)
    m = len(labels)
    masks = ((np.arange(1 << m)[:, None, None] >> mapping[None]) & 1).astype(bool)
    values = np.empty((len(x), len(background), 1 << m), dtype=np.float64)
    for i, sample in enumerate(xx):
        hybrids = np.where(masks[None], sample[None, None], bb[:, None])
        hybrids = hybrids.reshape(-1, *xx.shape[1:])
        if mode == 'frequency':
            hybrids = np.fft.irfft(hybrids, n=x.shape[1], axis=1)
        values[i] = probabilities(net, hybrids.astype(np.float32))[:, 1].reshape(len(background), -1)
        if (i + 1) % 5 == 0:
            print(f'{mode}: {i+1}/{len(x)}', flush=True)
    per_background = shapley(values, m)
    phi = per_background.mean(axis=1)
    importance = np.abs(phi).mean(axis=0)
    # Two balanced, disjoint reference sets show sensitivity to the background.
    halves = [np.abs(per_background[:, ids].mean(axis=1)).mean(axis=0)
              for ids in ([0, 1, 4, 5], [2, 3, 6, 7])]
    residual = np.max(np.abs(phi.sum(axis=1) - (values[:, :, -1].mean(axis=1) - values[:, :, 0].mean(axis=1))))
    assert residual < 1e-6, residual
    original = probabilities(net, x)[:, 1]
    assert np.max(np.abs(values[:, :, -1] - original[:, None])) < 2e-5
    return {'labels': labels, 'mean_abs_shap': importance.tolist(),
            'relative_importance': (importance / max(importance.sum(), 1e-12)).tolist(),
            'background_half_importance': [v.tolist() for v in halves],
            'phi': phi.tolist(), 'baseline': values[:, :, 0].mean(axis=1).tolist(),
            'predicted_wood02': original.tolist(), 'max_additivity_error': float(residual)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model_path = args.run / 'time/model.json'
    model = json.loads(model_path.read_text(encoding='utf-8'))
    assert model['config']['input_mode'] == 'time'
    net = CNN({**model['config'], 'channels': 3}, model['weights'])
    sources = json.loads((args.run / 'data_audit.json').read_text(encoding='utf-8'))['files']
    for source in sources:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    groups, time, y, _ = read_data([s['path'] for s in sources], model['config'])
    splits = json.loads((args.run / 'split_groups.json').read_text(encoding='utf-8'))
    ids = {g['id']: i for i, g in enumerate(groups)}
    test = np.array([ids[s] for s in splits['test']])
    train = np.array([ids[s] for s in splits['train']])
    assert not set(train) & set(test)
    rng = np.random.default_rng(20261018)
    bg = np.concatenate([rng.choice(train[y[train] == cls], 4, replace=False) for cls in range(2)])
    x = time[test]
    # Authoritative arithmetic check against a known additive coalition game.
    masks = np.array(list(range(8)))
    known = 0.25 + sum((masks >> j & 1) * v for j, v in enumerate([0.1, -0.3, 0.7]))
    assert np.allclose(shapley(known, 3), [0.1, -0.3, 0.7])
    result = {'model_version': model['model_version'],
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'method': 'Exact grouped interventional Shapley; empirical 8-triplet training background; output P(WOOD02)',
              'test_ids': [groups[i]['id'] for i in test], 'background_ids': [groups[i]['id'] for i in bg],
              'classes': model['config']['classes'], 'true_labels': y[test].tolist(), 'analyses': {}}
    for mode in ['position', 'time', 'frequency']:
        result['analyses'][mode] = audit(net, x, time[bg], mode)
        (args.out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    original = probabilities(net, x)
    probes = {}
    for factor in [0.5, 0.75, 1., 1.25, 1.5]:
        pp = probabilities(net, x * factor)
        probes[str(factor)] = {'correct': int((pp.argmax(1) == y[test]).sum()),
                              'count': len(test), 'flips': int((pp.argmax(1) != original.argmax(1)).sum()),
                              'mean_abs_probability_change': float(np.abs(pp[:, 1] - original[:, 1]).mean())}
    target_rms = np.median(np.sqrt(np.mean(time[train] ** 2, axis=1)), axis=0)
    normalized = x * (target_rms / np.maximum(np.sqrt(np.mean(x ** 2, axis=1)), 1e-12))[:, None, :]
    pp = probabilities(net, normalized)
    probes['equal_rms'] = {'correct': int((pp.argmax(1) == y[test]).sum()), 'count': len(test),
                           'flips': int((pp.argmax(1) != original.argmax(1)).sum()),
                           'target_rms_by_position': target_rms.tolist()}
    result['amplitude_probes'] = probes
    result['limitations'] = ['Only the two known specimens and existing capture conditions.',
        '20 held-out matching-H triplets / 60 unique hits, not 2000 independent combinations.',
        'Frozen deployment choice already used held-out results; this is retrospective interpretation.',
        'Eight balanced training references; exact coalition enumeration does not eliminate background uncertainty.',
        'Splicing time blocks or Fourier bands can create out-of-distribution signals.',
        'Attributions explain model scores under these interventions, not physical causality or the learning process.',
        'Quality gate is not included; amplitude probes evaluate classifier scores only.',
        'No new wood, reclamping, or independent hardware verification.']
    (args.out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    sections = ['# 当前时域模型的 SHAP 诊断', '', f"模型：{model['model_version']}。解释输出为 WOOD02 分数；正值推向木头2，负值推向木头1。", '',
        '使用原留出集的20组三点输入（60次互不重复敲击），参考来自训练集，每类4组。保留原始权重和正反方向平均推理。枚举全部特征组合，计算分组干预式 Shapley 值，无代理模型、无重新训练。', '']
    html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>木材模型 SHAP 诊断</title><style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1000px;margin:40px auto;background:#f6f8fb;color:#172338}section{background:white;padding:24px;margin:20px 0;border-radius:14px}h1{font-size:30px}p{line-height:1.8}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}</style><h1>当前时域模型：判断依据诊断</h1><p>20组留出三点输入 · 8组训练参考 · 精确枚举的分组 SHAP<br>比例仅代表该分组方案中的平均绝对贡献，不是准确率，也不是物理因果证据。</p>']
    for mode, title in [('position', '敲击位置'), ('time', '振动时间段（三个位置联合）'), ('frequency', '频带（三个位置联合，仍解释时域模型）')]:
        a = result['analyses'][mode]
        sections += ['## ' + title, '', '| 输入分组 | 平均绝对贡献（概率百分点） | 组内贡献占比 |', '|---|---:|---:|']
        html.append(f'<section><h2>{title}</h2><svg viewBox="0 0 940 {len(a["labels"])*49}" role="img" aria-label="{title}贡献">')
        for j in np.argsort(a['mean_abs_shap'])[::-1]:
            label, score, fraction = a['labels'][j], a['mean_abs_shap'][j], a['relative_importance'][j]
            sections.append(f'| {label} | {score*100:.2f} | {fraction:.1%} |')
        for row, j in enumerate(np.argsort(a['mean_abs_shap'])[::-1]):
            label, fraction = a['labels'][j], a['relative_importance'][j]
            html.append(f'<text x="0" y="{row*49+25}" font-size="16">{label}</text><rect x="270" y="{row*49+6}" width="{fraction*540:.1f}" height="28" rx="4" fill="#197b8c"/><text x="{280+fraction*540:.1f}" y="{row*49+26}" font-size="16">{fraction:.1%}</text>')
        html.append('</svg></section>')
        sections += ['', f"加和校验最大误差：{a['max_additivity_error']:.2e}。", '']
    sections += ['## 振幅干预测试', '', '| 操作 | 正确组数 | 相对原预测翻转 |', '|---|---:|---:|']
    html.append('<section><h2>振幅干预测试</h2><table><tr><th>操作</th><th>正确组数</th><th>预测翻转</th></tr>')
    for name, probe in probes.items():
        label = '每个位置统一到训练集的中位均方根振幅' if name == 'equal_rms' else f'全部波形振幅 ×{name}'
        sections.append(f'| {label} | {probe["correct"]}/{probe["count"]} | {probe["flips"]} |')
        html.append(f'<tr><td>{label}</td><td>{probe["correct"]}/{probe["count"]}</td><td>{probe["flips"]}</td></tr>')
    caution = '仅解释已知两根木头、既有采集条件下的模型判断，不能证明学到空心/实心或缺陷机理。时间拼接和频带替换可能制造真实采集中没有的信号；结果依赖参考数据、分组边界和输入相关性。振幅干预只测试分类器，不含前置质量拦截；分数变化不能直接等同真实敲击力度的效果。该模型部署选择已参考过留出结果，因此本报告是回顾性诊断，不是新增独立验收。'
    sections += ['', '## 解释边界', '', caution, '', 'results.json 保留逐样本有符号贡献、两套互不重叠且类别平衡的参考子集结果、样本编号、权重指纹及验证残差。']
    html.append(f'</table></section><section><h2>解释边界</h2><p>{caution}</p></section></html>')
    (args.out / 'report.md').write_text('\n'.join(sections), encoding='utf-8')
    (args.out / 'report.html').write_text(''.join(html), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'amplitude_probes': probes}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
