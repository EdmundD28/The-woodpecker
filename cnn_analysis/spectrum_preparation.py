"""Versioned, fitted three-point spectrum preparation used by deployed models."""
import numpy as np

VERSION='WOOD_SPECTRUM_RMS_LOG_V1'


def validate(preparation):
    if preparation.get('version')!=VERSION:
        raise ValueError('unsupported fitted spectrum preparation')
    for key in ['rms_target','log_reference','log_gain']:
        value=np.asarray(preparation.get(key),dtype=np.float64)
        if value.shape!=(3,) or not np.isfinite(value).all() or np.any(value<=0):
            raise ValueError('invalid spectrum '+key)
        if value[0]!=value[2]:
            raise ValueError('endpoint preparation must be symmetric')
    return preparation


def rms_log(time_inputs,c,preparation):
    validate(preparation)
    x=np.asarray(time_inputs,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(c['sample_count'],3) or not np.isfinite(x).all():
        raise ValueError('expected finite three-point time inputs')
    rms=np.sqrt(np.mean(x*x,axis=1,keepdims=True))
    normalized=(x*np.asarray(preparation['rms_target'])[None,None,:]/np.maximum(rms,1e-12)).astype(np.float32)
    windowed=normalized.astype(np.float64)*np.hanning(c['sample_count'])[None,:,None]
    magnitude=(np.abs(np.fft.rfft(windowed,axis=1))/c['sample_count']).astype(np.float32)
    return (np.log1p(magnitude.astype(np.float64)/np.asarray(preparation['log_reference'])[None,None,:])*
            np.asarray(preparation['log_gain'])[None,None,:]).astype(np.float32)
