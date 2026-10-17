#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifndef WOOD_USE_SPECTRUM
#define WOOD_USE_SPECTRUM 0
#endif

#if WOOD_USE_SPECTRUM
#include "models/spectrum_weights.h"
#else
#include "models/time_weights.h"
#endif

namespace wood_classifier {
constexpr size_t sample_count=2400;
constexpr uint32_t sample_rate_hz=8000;
constexpr size_t point_length=WOOD_USE_SPECTRUM ? 1201 : 2400;
constexpr size_t scratch_count=WOOD_USE_SPECTRUM ? 9664 : 19200;
#if WOOD_USE_SPECTRUM
constexpr char mode[]="SPECTRUM";
constexpr char version[]="REAL-TRIPLET-f392163987d7";
#else
constexpr char mode[]="TIME";
constexpr char version[]="REAL-TRIPLET-beecd884647a";
#endif

// One interleaved triplet, reused across groups. No heap allocation at inference.
static float input[point_length*3];
// ESP32's statically addressable DRAM is smaller than its total internal RAM.
// Reserve the large scratch buffer once at boot from internal heap instead.
static float* scratch=nullptr;

#if WOOD_USE_SPECTRUM
struct Complex { float re, im; };
static Complex fft_input[sample_count], fft_output[sample_count], roots[sample_count];

inline bool initialise() {
    if(!scratch) scratch=static_cast<float*>(std::malloc(scratch_count*sizeof(float)));
    if(!scratch) return false;
    constexpr double pi=3.14159265358979323846;
    for(size_t k=0;k<sample_count;++k) {
        double angle=-2*pi*k/sample_count;
        roots[k]={float(std::cos(angle)),float(std::sin(angle))};
    }
    return true;
}

// Exact 2400-point mixed-radix FFT (2*2*2*2*2*3*5*5), without zero padding.
// Each recursive output occupies a disjoint contiguous region; radix <= 5.
inline void fft(const Complex* source,Complex* output,size_t n,size_t stride) {
    if(n==1) { output[0]=*source; return; }
    size_t radix=n%2==0?2:n%3==0?3:5;
    size_t m=n/radix;
    for(size_t j=0;j<radix;++j) fft(source+j*stride,output+j*m,m,stride*radix);
    for(size_t k=0;k<m;++k) {
        Complex values[5];
        for(size_t q=0;q<radix;++q) {
            values[q]={0,0};
            for(size_t j=0;j<radix;++j) {
                const Complex a=output[j*m+k];
                const Complex w=roots[(j*(k+q*m)*stride)%sample_count];
                values[q].re+=a.re*w.re-a.im*w.im;
                values[q].im+=a.re*w.im+a.im*w.re;
            }
        }
        for(size_t q=0;q<radix;++q) output[q*m+k]=values[q];
    }
}
#else
inline bool initialise() {
    if(!scratch) scratch=static_cast<float*>(std::malloc(scratch_count*sizeof(float)));
    return scratch!=nullptr;
}
#endif

inline bool store_point(const uint16_t* raw,size_t point) {
    if(point>=3) return false;
    uint32_t sum=0;
    for(size_t i=0;i<sample_count;++i) sum+=raw[i];
    float mean=float(sum)/float(sample_count);
#if WOOD_USE_SPECTRUM
    for(size_t i=0;i<sample_count;++i) {
        float time=(float(raw[i])-mean)/2048.0f;
        fft_input[i]={float(double(time)*wood_triplet_spectrum::hann[i]),0};
    }
    fft(fft_input,fft_output,sample_count,1);
    for(size_t k=0;k<point_length;++k) {
        float re=fft_output[k].re, im=fft_output[k].im;
        float value=std::sqrt(re*re+im*im)/float(sample_count);
        if(!std::isfinite(value)) return false;
        input[k*3+point]=value;
    }
#else
    for(size_t i=0;i<sample_count;++i) input[i*3+point]=(float(raw[i])-mean)/2048.0f;
#endif
    return true;
}

inline void reverse_endpoints() {
    for(size_t i=0;i<point_length;++i) {
        float value=input[i*3]; input[i*3]=input[i*3+2]; input[i*3+2]=value;
    }
}

// The training contract averages forward and reversed softmax scores.
// Swap in place to save a second 14/28 KB triplet buffer, then restore it.
inline bool classify(float* scores) {
    float reversed[2];
#if WOOD_USE_SPECTRUM
    wood_triplet_spectrum::infer(input,scores,scratch);
    reverse_endpoints();
    wood_triplet_spectrum::infer(input,reversed,scratch);
#else
    wood_triplet_time::infer(input,scores,scratch);
    reverse_endpoints();
    wood_triplet_time::infer(input,reversed,scratch);
#endif
    reverse_endpoints();
    for(size_t p=0;p<2;++p) {
        scores[p]=(scores[p]+reversed[p])*0.5f;
        if(!std::isfinite(scores[p])) return false;
    }
    return true;
}
} // namespace wood_classifier
