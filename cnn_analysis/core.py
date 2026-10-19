"""Strict data adapter, shared preprocessing and leakage-safe split."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from pc_collector.protocol import validate_record

ROOT = Path(__file__).resolve().parent


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def config(path=None):
    c = json.loads(Path(path or ROOT/'config.json').read_text(encoding='utf-8-sig'))
    return validate_config(c)


def validate_config(c):
    profiles = {'TIME_V0_1': (16000, 1024), 'TIME_V0_2': (8000, 2400)}
    version = c.get('interface_version')
    if version not in profiles:
        raise ValueError('unsupported preprocessing interface version')
    fs, count = profiles[version]
    fixed = {'sample_rate_hz':fs, 'sample_count':count, 'channels':1,
             'scale_divisor':2048.0, 'trigger_index':256}
    if any(c.get(k) != v for k,v in fixed.items()):
        raise ValueError(version + ' fixed interface mismatch')
    if c.get('split_strategy', 'group') not in ['group', 'stratified_hit']:
        raise ValueError('split_strategy must be group or stratified_hit')
    if c.get('input_mode', 'time') not in ['time','spectrum']:
        raise ValueError('input_mode must be time or spectrum')
    if 'spectrum_preparation' in c:
        from .spectrum_preparation import validate
        if c.get('input_mode')!='spectrum':
            raise ValueError('fitted spectrum preparation requires spectrum mode')
        validate(c['spectrum_preparation'])
    if len(c['classes']) != 2 or len(set(c['classes'])) != 2:
        raise ValueError('10/09 scope requires exactly two distinct classes')
    if len(c['class_display_names']) != 2:
        raise ValueError('need two display names in class order')
    if any(v not in c['classes'] for v in c['stick_labels'].values()):
        raise ValueError('stick_labels contains an unknown class')
    for key in ['epochs','batch_size','patience']:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError('invalid ' + key)
    if not np.isfinite(c['learning_rate']) or c['learning_rate'] <= 0:
        raise ValueError('invalid learning_rate')
    if c['conv_layers'] != [[8,9,2],[16,5,2],[32,3,2]]:
        raise ValueError('this version exports the three-layer candidate CNN only')
    threshold = c['confidence_threshold']
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('invalid confidence_threshold')
    if not (0 <= c['clip_low'] < c['clip_high'] <= 4095) or c['min_peak_to_peak'] < 0:
        raise ValueError('invalid quality thresholds')
    return c


def preprocess(record, c):
    validate_record(record)
    if record['sample_rate_hz'] != c['sample_rate_hz'] or record['sample_count'] != c['sample_count']:
        raise ValueError(f"expected {c['sample_rate_hz']} Hz and {c['sample_count']} points; no resampling/padding/truncation")
    if record['anomaly_flags']:
        raise ValueError('anomaly_flags=' + ','.join(record['anomaly_flags']))
    raw = np.asarray(record['raw'], dtype=np.int64)
    if np.any(raw < 0) or np.any(raw > 4095):
        raise ValueError('ADC out of range')
    if np.any(raw <= c['clip_low']) or np.any(raw >= c['clip_high']):
        raise ValueError('CLIPPED detected independently')
    if int(np.ptp(raw)) < c['min_peak_to_peak']:
        raise ValueError('WEAK_SIGNAL detected independently')
    # Sum integer ADC values first: exactly matches firmware B mean/divisor.
    mean = np.float32(int(raw.sum()) / c['sample_count'])
    x = (raw.astype(np.float32) - mean) / np.float32(c['scale_divisor'])
    return x.reshape(c['sample_count'],1)


def model_input(time_inputs,c):
    """Convert the complete configured time window to model input.

    Symmetric Hann (denominator N-1), RFFT magnitude / N. No log transform,
    peak normalization, phase, bin dropping or one-sided doubling.
    Accepts [batch,sample_count,1]; bins include DC and Nyquist.
    """
    count = c['sample_count']
    if time_inputs.ndim != 3 or time_inputs.shape[1:] != (count,1):
        raise ValueError(f"expected {c['interface_version']} batch [batch,{count},1]")
    if c.get('input_mode','time') == 'time':
        return time_inputs.astype(np.float32,copy=False)
    windowed = time_inputs[:,:,0].astype(np.float64)*np.hanning(count)
    return (np.abs(np.fft.rfft(windowed,axis=1))/count).astype(np.float32)[:,:,None]


def input_contract(c):
    count, fs = c['sample_count'], c['sample_rate_hz']
    version = c['interface_version']
    if c.get('input_mode','time') == 'time':
        return {'version':version,'shape':[count,1],'dtype':'float32',
                'operation':'whole-window mean removal, fixed /2048; no FFT'}
    return {'version':version.replace('TIME_', 'SPECTRUM_'),'shape':[count//2+1,1],'dtype':'float32',
            'base_preprocessing':version, 'window':f'symmetric Hann, denominator {count-1}',
            'operation':f'abs(rfft(time_input * hann))/{count}',
            'bin_range':f'0..{count//2} inclusive (DC through Nyquist)',
            'bin_spacing_hz':fs/count,'phase':False,'one_sided_doubling':False,
            'log_transform':False,'per_record_peak_normalization':False}


def read_manifest(path):
    if path is None:
        return {}
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        required={'sample_id','label','source','release_method','electrical_verified'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('manifest headers must include '+','.join(sorted(required)))
        for row in reader:
            if None in row:
                raise ValueError('manifest row has extra columns')
            sid = row.get('sample_id', '').strip()
            if not sid or sid in result:
                raise ValueError('manifest empty or duplicate sample_id')
            result[sid] = {k:(v or '').strip() for k,v in row.items()}
    return result


def load(paths, c, manifest=None):
    files = []
    for item in paths:
        p = Path(item)
        if not p.exists():
            raise ValueError('input does not exist: ' + str(p))
        files.extend(sorted(q for q in p.rglob('*') if q.suffix.lower() in ['.json','.jsonl']) if p.is_dir() else [p])
    files = sorted(set(p.resolve() for p in files))
    meta = read_manifest(manifest)
    accepted, rejected, duplicates = [], [], []
    seen = {}
    for p in files:
        with p.open(encoding='utf-8-sig') as f:
            if p.suffix.lower() == '.json':
                entries = [(1, f.read())]
            else:
                entries = enumerate(f, 1)
            for line, text in entries:
                if not text.strip():
                    continue
                locator = {'file':str(p), 'line':line}
                try:
                    r = json.loads(text)
                    validate_record(r)
                    sid = r['sample_id']
                    fingerprint = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                    if sid in seen:
                        if seen[sid] != fingerprint:
                            raise RuntimeError('conflicting records with same sample_id: ' + sid)
                        duplicates.append({**locator, 'sample_id':sid})
                        continue
                    seen[sid] = fingerprint
                    x = preprocess(r, c)
                    m = meta.get(sid, {})
                    label = m.get('label') or c['stick_labels'].get(r['stick_id'])
                    if label and label not in c['classes']:
                        raise ValueError('unknown class label')
                    mapped = c['stick_labels'].get(r['stick_id'])
                    if mapped and label != mapped:
                        raise ValueError('manifest/config label conflict')
                    source = m.get('source') or 'unknown'
                    if source not in ['real','synthetic','unknown']:
                        raise ValueError('source must be real or synthetic')
                    release = m.get('release_method') or 'unknown'
                    if release not in ['manual','mechanical','unknown']:
                        raise ValueError('release_method must be manual or mechanical')
                    accepted.append({'record':r, 'input':x, 'label':label, 'source':source,
                                     'release_method':release,
                                     'electrical_verified':m.get('electrical_verified','').lower() == 'true',
                                     'group':f"E{r['experiment_batch']:02}-R{r['reclamp_batch']:02}", **locator})
                except (ValueError, TypeError, KeyError, OverflowError) as e:
                    rejected.append({**locator, 'reason':str(e)})
    # Different labels for the same physical stick indicate inconsistent metadata.
    labels = {}
    for a in accepted:
        stick = a['record']['stick_id']
        if a['label']:
            if stick in labels and labels[stick] != a['label']:
                raise ValueError('inconsistent labels for stick: ' + stick)
            labels[stick] = a['label']
    return accepted, rejected, duplicates


def split(records, c):
    if not records or any(a['label'] not in c['classes'] for a in records):
        raise ValueError('all accepted records need confirmed class labels')
    if c.get('split_strategy', 'group') == 'stratified_hit':
        return split_hits(records, c)
    groups = sorted({a['group'] for a in records})
    if len(groups) < 5:
        raise ValueError('need >=5 independent E/R groups; never substitute a random hit split')
    rng = np.random.default_rng(c['seed'])
    n = max(1, int(len(groups)*.2))
    for _ in range(1000):
        order = rng.permutation(groups).tolist()
        partition = {'test':order[:n], 'validation':order[n:2*n], 'train':order[2*n:]}
        ids = {name:np.array([i for i,a in enumerate(records) if a['group'] in gs],dtype=int)
               for name,gs in partition.items()}
        if all({records[i]['label'] for i in ix} == set(c['classes']) for ix in ids.values()):
            return ids, partition
    raise ValueError('cannot make group-disjoint partitions containing both classes; collect balanced batches')


def split_hits(records, c):
    """Hold out independent hits on the same known boards; never split a hit."""
    sample_ids = [a['record']['sample_id'] for a in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('duplicate sample_id; deduplicate independent hits before splitting')
    rng = np.random.default_rng(c['seed'])
    parts = {'train': [], 'validation': [], 'test': []}
    for label in c['classes']:
        indices = np.array([i for i,a in enumerate(records) if a['label'] == label], dtype=int)
        if len(indices) < 3:
            raise ValueError('stratified_hit requires >=3 independent hits per class')
        indices = rng.permutation(indices)
        held = max(1, int(len(indices) * .2))
        parts['test'].extend(indices[:held])
        parts['validation'].extend(indices[held:2*held])
        parts['train'].extend(indices[2*held:])
    ids = {name: rng.permutation(np.asarray(ix, dtype=int)) for name,ix in parts.items()}
    partition = {name: [sample_ids[i] for i in ix] for name,ix in ids.items()}
    return ids, partition


def dataset_hash(records):
    payload = [{'record':a['record'], 'label':a['label'], 'source':a['source'],
                'release_method':a['release_method'], 'electrical_verified':a['electrical_verified']}
               for a in sorted(records,key=lambda a:a['record']['sample_id'])]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
