#pragma once
#include "Arduino.h"
#include "adc.h"
using i2s_port_t=int; using i2s_mode_t=int; using QueueHandle_t=void*;
constexpr int I2S_NUM_0=0,I2S_MODE_MASTER=1,I2S_MODE_RX=2,I2S_MODE_ADC_BUILT_IN=4;
constexpr int I2S_BITS_PER_SAMPLE_16BIT=16,I2S_CHANNEL_FMT_ONLY_LEFT=1;
constexpr int I2S_COMM_FORMAT_STAND_MSB=1,ESP_INTR_FLAG_LEVEL1=1,pdTRUE=1;
constexpr int I2S_EVENT_RX_Q_OVF=1,I2S_EVENT_DMA_ERROR=2;
struct i2s_config_t {
    int mode,sample_rate,bits_per_sample,channel_format,communication_format;
    int intr_alloc_flags,dma_buf_count,dma_buf_len; bool use_apll;
};
struct i2s_event_t { int type; };
inline std::deque<int> fakeEvents;
inline std::deque<uint16_t> fakeAdc;
inline int xQueueReceive(QueueHandle_t,i2s_event_t* event,int) {
    if(fakeEvents.empty()) return 0;
    event->type=fakeEvents.front(); fakeEvents.pop_front(); return 1;
}
inline void xQueueReset(QueueHandle_t) { fakeEvents.clear(); }
inline int pdMS_TO_TICKS(int n) { return n; }
inline int i2s_driver_install(int,const i2s_config_t*,int,QueueHandle_t* q) { *q=(void*)1; return 0; }
inline int i2s_set_adc_mode(int,int) { return 0; }
inline int i2s_adc_enable(int) { return 0; }
inline int i2s_adc_disable(int) { return 0; }
inline int i2s_start(int) { return 0; }
inline int i2s_stop(int) { return 0; }
inline int i2s_read(int,void* dest,size_t capacity,size_t* bytes,int) {
    size_t n=std::min(capacity/2,fakeAdc.size());
    auto out=static_cast<uint16_t*>(dest);
    for(size_t i=0;i<n;++i) { out[i]=fakeAdc.front(); fakeAdc.pop_front(); }
    *bytes=n*2; return 0;
}
