#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "models/quality_reference.h"

// One single-hit quality entry point. No learning or threshold adaptation on device.
namespace signal_quality {
constexpr size_t N=2400, PRE=256, POST=N-PRE, FEATURES=27;
constexpr size_t workspace_floats=4*POST; // Borrow CNN scratch only while inference is idle.
enum class Policy { Strict, Venue };
inline float multiplier(Policy p) { return p==Policy::Venue?1.5f:1.0f; }
inline const char* policyName(Policy p) { return p==Policy::Venue?"VENUE":"STRICT"; }
struct Result {
    const char* problem=nullptr;
    float score=0, baseThreshold=0, threshold=0;
    bool passed() const { return problem==nullptr; }
};
inline const char* basic(const uint16_t* raw,size_t count,bool lost) {
    if(lost || count!=N) return "SAMPLE_LOSS";
    uint16_t low=4095,high=0;
    for(size_t i=0;i<N;++i) {
        if(raw[i]<=10 || raw[i]>=4085) return "CLIPPED";
        if(raw[i]<low) low=raw[i]; if(raw[i]>high) high=raw[i];
    }
    return high-low<100?"WEAK_SIGNAL":nullptr;
}

// Exact length 2144 = 2^5 * 67. Arrays are interleaved complex floats.
// The largest local radix is 67; no heap allocations or 4096-point padding.
inline void fft(const float* source,float* output,size_t n,size_t stride) {
    if(n==1) { output[0]=source[0]; output[1]=source[1]; return; }
    const size_t radix=n%2==0?2:67;
    const size_t m=n/radix;
    for(size_t j=0;j<radix;++j) fft(source+2*j*stride,output+2*j*m,m,stride*radix);
    for(size_t k=0;k<m;++k) {
        float re[67],im[67];
        for(size_t q=0;q<radix;++q) {
            double sumRe=0,sumIm=0;
            for(size_t j=0;j<radix;++j) {
                const float* a=output+2*(j*m+k);
                const float* w=quality_reference::roots[(j*(k+q*m)*stride)%POST];
                sumRe+=double(a[0])*w[0]-double(a[1])*w[1];
                sumIm+=double(a[0])*w[1]+double(a[1])*w[0];
            }
            re[q]=float(sumRe); im[q]=float(sumIm);
        }
        for(size_t q=0;q<radix;++q) {
            output[2*(q*m+k)]=re[q]; output[2*(q*m+k)+1]=im[q];
        }
    }
}

inline bool features(const uint16_t* raw,float* feature,float* workspace) {
    if(!workspace) return false;
    uint32_t sum=0; for(size_t i=0;i<N;++i) sum+=raw[i];
    const float mean=float(sum)/float(N);
    double preEnergy=0,postEnergy=0,peak=0,temporal[8]={},bands[12]={};
    float low=1e9f,high=-1e9f;
    float* input=workspace; float* transformed=workspace+2*POST;
    for(size_t i=0;i<N;++i) {
        const float x=(float(raw[i])-mean)/2048.0f;
        if(x<low) low=x; if(x>high) high=x;
        const double e=double(x)*x;
        if(i<PRE) preEnergy+=e;
        else {
            const size_t j=i-PRE;
            postEnergy+=e; temporal[j/268]+=e;
            if(std::fabs(double(x))>peak) peak=std::fabs(double(x));
            input[2*j]=x*quality_reference::hann[j]; input[2*j+1]=0;
        }
    }
    const double total=postEnergy>1e-12?postEnergy:1e-12;
    for(size_t b=0;b<8;++b) feature[b]=float(std::log(temporal[b]/total+1e-6));
    fft(input,transformed,POST,1);
    size_t bin=1; double allPower=0;
    for(size_t b=0;b<12;++b) {
        const size_t count=b<4?90:89; // np.array_split of 1072 positive-frequency bins.
        for(size_t j=0;j<count;++j,++bin) {
            const double re=transformed[2*bin],im=transformed[2*bin+1];
            bands[b]+=re*re+im*im;
        }
        allPower+=bands[b];
    }
    if(allPower<1e-12) allPower=1e-12;
    for(size_t b=0;b<12;++b) feature[8+b]=float(std::log(bands[b]/allPower+1e-6));
    double cumulative=0; unsigned quantile=0; const double levels[3]={.1,.5,.9};
    for(size_t j=0;j<POST && quantile<3;++j) {
        const float x=(float(raw[j+PRE])-mean)/2048.0f;
        cumulative+=double(x)*x;
        while(quantile<3 && cumulative/total>=levels[quantile]) feature[20+quantile++]=float(j)/float(POST);
    }
    while(quantile<3) feature[20+quantile++]=1.0f;
    const double whole=preEnergy+postEnergy;
    feature[23]=float(preEnergy/(whole>1e-12?whole:1e-12));
    const double rms=std::sqrt(postEnergy/POST);
    feature[24]=float(std::log((peak>1e-9?peak:1e-9)/(rms>1e-9?rms:1e-9)));
    const double range=double(high)-low;
    feature[25]=float(std::log(range>1e-9?range:1e-9));
    feature[26]=float(std::log(rms>1e-9?rms:1e-9));
    for(size_t i=0;i<FEATURES;++i) if(!std::isfinite(feature[i])) return false;
    return true;
}

inline float distance(const float* feature,bool middle,Policy policy=Policy::Strict) {
    const float* median=middle?quality_reference::middle_median:quality_reference::endpoint_median;
    const float* scale=middle?quality_reference::middle_scale:quality_reference::endpoint_scale;
    const float* reference=middle?&quality_reference::middle_reference[0][0]:&quality_reference::endpoint_reference[0][0];
    const size_t count=middle?quality_reference::middle_count:quality_reference::endpoint_count;
    const size_t dimensions=policy==Policy::Venue?quality_reference::venue_feature_count:FEATURES;
    float z[FEATURES]; for(size_t i=0;i<FEATURES;++i) z[i]=(feature[i]-median[i])/scale[i];
    double best[5]={1e100,1e100,1e100,1e100,1e100};
    for(size_t row=0;row<count;++row) {
        double total=0;
        for(size_t i=0;i<dimensions;++i) { const double d=z[i]-reference[row*FEATURES+i]; total+=d*d; }
        const double value=std::sqrt(total/dimensions);
        for(size_t k=0;k<5;++k) if(value<best[k]) {
            for(size_t j=4;j>k;--j) best[j]=best[j-1]; best[k]=value; break;
        }
    }
    double total=0; for(double value:best) total+=value;
    return float(total/5);
}

// slot=0 endpoint, 1 middle, 2 opposite endpoint: both 123 and 321 use the same gate.
inline Result evaluate(const uint16_t* raw,size_t count,bool lost,size_t slot,
                       Policy policy,float* workspace) {
    Result r; r.problem=basic(raw,count,lost);
    if(r.problem) return r;
    if(slot>=3 || !workspace) { r.problem="QUALITY_ERROR"; return r; }
    float f[FEATURES];
    if(!features(raw,f,workspace)) { r.problem="QUALITY_ERROR"; return r; }
    r.score=distance(f,slot==1,policy);
    r.baseThreshold=policy==Policy::Venue?
        (slot==1?quality_reference::middle_venue_threshold:quality_reference::endpoint_venue_threshold):
        (slot==1?quality_reference::middle_threshold:quality_reference::endpoint_threshold);
    r.threshold=r.baseThreshold*multiplier(policy);
    if(!std::isfinite(r.score)) r.problem="QUALITY_ERROR";
    else if(r.score>r.threshold) r.problem="SHAPE_OUTLIER";
    return r;
}
} // namespace signal_quality
