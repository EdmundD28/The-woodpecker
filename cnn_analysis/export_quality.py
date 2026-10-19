"""Export the frozen normal-only quality reference and exact 2144-point FFT constants."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import copy


def venue_model(model_path):
    """Drop absolute amplitude/RMS, recalibrate only on the saved normal calibration pool."""
    from .quality import calibrate
    path=Path(model_path)
    model=copy.deepcopy(json.loads(path.read_text(encoding='utf-8')))
    model['feature_names']=model['feature_names'][:25]
    for g in model['groups'].values():
        g['median']=g['median'][:25]; g['scale']=g['scale'][:25]
        g['reference']=[row[:25] for row in g['reference']]
    normal=[json.loads(s) for s in (path.parent/'calibration.jsonl').read_text(encoding='utf-8').splitlines()]
    return calibrate(model,normal)


def export(model_path, output):
    path=Path(model_path); payload=path.read_bytes(); model=json.loads(payload)
    if model['format_version']!='WOOD_NORMAL_QUALITY_V1': raise ValueError('unsupported model')
    c=model['config']
    if model['k']!=5 or len(model['feature_names'])!=27:
        raise ValueError('firmware expects the 27-feature, five-neighbor reference')
    venue=venue_model(path)
    if (c['sample_rate_hz'],c['sample_count'],c['trigger_index'])!=(8000,2400,256):
        raise ValueError('quality firmware expects 8kHz/2400/256')
    n=2144
    lines=['#pragma once','// Generated from frozen normal-only model; do not edit arrays.',
           'namespace quality_reference {',f'constexpr char sha256[]="{hashlib.sha256(payload).hexdigest()}";',
           'constexpr unsigned feature_count=27, venue_feature_count=25, neighbor_count=5;']
    def array(name,value):
        a=np.asarray(value,dtype=np.float32)
        dims=''.join(f'[{d}]' for d in a.shape)
        lines.append(f'static const float {name}{dims}={{'+','.join(f'{v:.9e}f' for v in a.ravel())+'};')
    for name,g in model['groups'].items():
        array(name+'_median',g['median']); array(name+'_scale',g['scale'])
        array(name+'_reference',g['reference'])
        lines.append(f'constexpr unsigned {name}_count={len(g["reference"])};')
        lines.append(f'constexpr float {name}_threshold={g["threshold"]:.9e}f;')
        lines.append(f'constexpr float {name}_venue_threshold={venue["groups"][name]["threshold"]:.9e}f;')
    array('hann',np.hanning(n))
    a=-2*np.pi*np.arange(n)/n
    array('roots',np.stack([np.cos(a),np.sin(a)],axis=1))
    lines.append('} // namespace quality_reference\n')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')
    result={'source_model_sha256':hashlib.sha256(payload).hexdigest(),
            'feature_count':27,'reference_count':sum(len(g['reference']) for g in model['groups'].values()),
            'reference_and_scaling_float_bytes':(sum(len(g['reference']) for g in model['groups'].values())*27+4*27)*4,
            'fft_constants_float_bytes':n*3*4,'workspace_bytes':n*4*4,
            'strict_features':27,'venue_features':25,'venue_threshold_multiplier':1.5,
            'venue_base_thresholds':{name:g['threshold'] for name,g in venue['groups'].items()},
            'venue_calibration':'same 58 normal calibration records only; no abnormal fitting',
            'selection_limit':'venue profile designed after synthetic perturbation audit; existing holdouts are development evidence, not fresh validation'}
    output.with_name('quality_deployment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('output'); a=p.parse_args()
    print(json.dumps(export(a.model,a.output),indent=2))
