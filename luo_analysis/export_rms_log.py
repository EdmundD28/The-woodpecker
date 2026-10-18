"""Patch the three-point C++ reference with the fitted RMS/log input contract."""
import re
from .spectrum_preparation import validate


def patch_header(header,model):
    c=model['config'];p=validate(c['spectrum_preparation']);n=c['sample_count'];bins=n//2+1
    constants='\n#define WOOD_SPECTRUM_RMS_LOG 1\n'
    constants+=f'static constexpr const char model_version[]="{model["model_version"]}";\n'
    for key in ['rms_target','log_reference','log_gain']:
        constants+=f'static const double {key}[3]={{'+','.join(f'{v:.17e}' for v in p[key])+'};\n'
    function=f'''inline void preprocess_point(const uint16_t* raw,float* input,int point) {{
    uint32_t sum=0; for(int i=0;i<{n};++i) sum+=raw[i];
    float mean=float(sum)/float({n});
    double energy=0;
    for(int i=0;i<{n};++i) {{ float t=(float(raw[i])-mean)/2048.0f; energy+=double(t)*double(t); }}
    double denominator=std::sqrt(energy/{n}.0);
    if(denominator<1e-12) denominator=1e-12;
    const double pi=3.14159265358979323846;
    for(int k=0;k<{bins};++k) {{
        double re=0,im=0;
        for(int i=0;i<{n};++i) {{
            float t=(float(raw[i])-mean)/2048.0f;
            float normalized=float(double(t)*rms_target[point]/denominator);
            double value=double(normalized)*hann[i];
            double angle=2*pi*k*i/{n}.0;
            re+=value*std::cos(angle); im-=value*std::sin(angle);
        }}
        float magnitude=float(std::sqrt(re*re+im*im)/{n}.0);
        input[k]=float(std::log1p(double(magnitude)/log_reference[point])*log_gain[point]);
    }}
}}
'''
    header,count=re.subn(r'inline void preprocess\(const uint16_t\* raw,float\* input\) \{.*?\n\}',constants+function,header,flags=re.S)
    if count!=1: raise ValueError('expected exactly one reference preprocessing function')
    header=header.replace('preprocess(points[p],point_scratch);','preprocess_point(points[p],point_scratch,p);')
    return header
