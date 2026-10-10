"""Export a self-contained float32 C++ reference, without touching firmware B."""
from pathlib import Path
import numpy as np
from .core import dump


def export(model,out):
    out = Path(out)
    weights = model['weights']
    arrays = []
    for name,value in weights.items():
        flat = np.asarray(value,dtype=np.float32).ravel()
        values = ','.join(f'{float(v):.9e}f' for v in flat)
        arrays.append(f'static const float {name}[{len(flat)}] = {{{values}}};')
    header = '''#pragma once
// Generated weights: inspect model_card.md before any deployment.
#include <cmath>
#include <cstdint>
namespace wood_model {
'''+ '\n'.join(arrays) + '''
// Conv SAME padding: floor(total_padding/2) on the left; kernel order [K,Cin,Cout].
inline void conv(const float* input, int len, int cin, float* output,
                 int cout, int kernel, int stride, const float* w, const float* b) {
    const int olen=(len+stride-1)/stride;
    const int total=(olen-1)*stride+kernel-len;
    const int left=total>0?total/2:0;
    for(int t=0;t<olen;++t) for(int o=0;o<cout;++o) {
        float value=b[o];
        for(int k=0;k<kernel;++k) {
            int at=t*stride+k-left;
            if(at<0 || at>=len) continue;
            for(int c=0;c<cin;++c) value+=input[at*cin+c]*w[(k*cin+c)*cout+o];
        }
        output[t*cout+o]=value>0?value:0;
    }
}
// Caller owns scratch[8192] (32 KiB); function is reentrant with separate scratch.
inline void infer(const float* input, float* scores, float* scratch) {
    float* a=scratch; float* b=scratch+4096;
    conv(input,1024,1,a,8,9,2,w0,b0);
    conv(a,512,8,b,16,5,2,w1,b1);
    conv(b,256,16,a,32,3,2,w2,b2);
    float pooled[32]={};
    for(int t=0;t<128;++t) for(int c=0;c<32;++c) pooled[c]+=a[t*32+c]/128.0f;
    for(int o=0;o<2;++o) {
        scores[o]=bd[o];
        for(int c=0;c<32;++c) scores[o]+=pooled[c]*wd[c*2+o];
    }
    float peak=scores[0]>scores[1]?scores[0]:scores[1];
    float sum=0;
    for(int o=0;o<2;++o) { scores[o]=std::exp(scores[o]-peak); sum+=scores[o]; }
    for(int o=0;o<2;++o) scores[o]/=sum;
}
// Quality checks remain the caller's responsibility; this only implements TIME_V0_1.
inline void preprocess(const uint16_t* raw,float* input) {
    uint32_t sum=0; for(int i=0;i<1024;++i) sum+=raw[i];
    float mean=float(sum)/1024.0f;
    for(int i=0;i<1024;++i) input[i]=(float(raw[i])-mean)/2048.0f;
}
} // namespace wood_model
'''
    mode=model['config'].get('input_mode','time')
    lengths=[1024 if mode=='time' else 513]
    channels=[1]
    for cout,kernel,stride in model['config']['conv_layers']:
        lengths.append((lengths[-1]+stride-1)//stride)
        channels.append(cout)
    buffer=max(lengths[i]*channels[i] for i in range(1,4))
    header=header.replace('scratch[8192] (32 KiB)',f'scratch[{buffer*2}] ({buffer*8} bytes)')
    header=header.replace('scratch+4096',f'scratch+{buffer}')
    header=header.replace('conv(input,1024,1,a,8,9,2,w0,b0);',f'conv(input,{lengths[0]},1,a,8,9,2,w0,b0);')
    header=header.replace('conv(a,512,8,b,16,5,2,w1,b1);',f'conv(a,{lengths[1]},8,b,16,5,2,w1,b1);')
    header=header.replace('conv(b,256,16,a,32,3,2,w2,b2);',f'conv(b,{lengths[2]},16,a,32,3,2,w2,b2);')
    header=header.replace('t<128',f't<{lengths[3]}').replace('/128.0f',f'/{lengths[3]}.0f')
    if mode=='spectrum':
        pool_start=header.index('    float pooled[32]={};')
        pool_end=header.index('    for(int o=0;o<2;++o)',pool_start)
        header=header[:pool_start]+'    const float* pooled=a; // Flatten preserves frequency-bin positions.\n'+header[pool_end:]
        header=header.replace('for(int c=0;c<32;++c) scores',f'for(int c=0;c<{lengths[3]*32};++c) scores')
        window=','.join(f'{float(v):.17e}' for v in np.hanning(1024))
        spectral='''// SPECTRUM_V0_1: direct DFT reference; replace with a verified FFT for hardware speed.
// Input is 1024 ADC points, output is float32[513], including DC and Nyquist.
static const double hann[1024] = {WINDOW};
inline void preprocess(const uint16_t* raw,float* input) {
    uint32_t sum=0; for(int i=0;i<1024;++i) sum+=raw[i];
    float mean=float(sum)/1024.0f;
    const double pi=3.14159265358979323846;
    for(int k=0;k<513;++k) {
        double re=0,im=0;
        for(int i=0;i<1024;++i) {
            float time=(float(raw[i])-mean)/2048.0f;
            double value=double(time)*hann[i];
            double angle=2.0*pi*k*i/1024.0;
            re+=value*std::cos(angle); im-=value*std::sin(angle);
        }
        input[k]=float(std::sqrt(re*re+im*im)/1024.0);
    }
}
'''.replace('WINDOW',window)
        start=header.index('// Quality checks remain')
        header=header[:start]+spectral+'} // namespace wood_model\n'
    header=header.replace('namespace wood_model',f'namespace wood_model_{mode}')
    (out/'model_weights.h').write_text(header,encoding='utf-8')
    count = sum(np.asarray(v).size for v in weights.values())
    dense_inputs=lengths[3]*32 if mode=='spectrum' else 32
    macs=sum(lengths[i+1]*cout*kernel*channels[i] for i,(cout,kernel,stride) in enumerate(model['config']['conv_layers']))+dense_inputs*2
    dump(out/'deployment_budget.json',{'input_mode':mode,'parameters':count,'float32_weight_bytes':count*4,
         'input_bytes':lengths[0]*4,'scratch_bytes':buffer*8,'local_pooled_bytes':0 if mode=='spectrum' else 128,
         'classification_head':'flatten' if mode=='spectrum' else 'global_average',
         'conv_dense_macs':macs,'spectrum_hann_reference_bytes':8192 if mode=='spectrum' else 0,
         'frequency_transform':'direct DFT reference (1024 x 513 iterations); FFT replacement required for practical latency' if mode=='spectrum' else 'none',
         'not_included':'ADC/DMA/OLED, runtime stack, code, allocator and firmware globals',
         'hardware_latency':'NOT_MEASURED','esp32_integration':'PENDING'})
