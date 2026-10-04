#pragma once
using esp_err_t=int;
constexpr int ESP_OK=0,ADC_WIDTH_BIT_12=12,ADC_ATTEN_DB_12=12,ADC_UNIT_1=1;
using adc1_channel_t=int;
constexpr int ADC1_CHANNEL_6=6;
inline int adc1_config_width(int) { return 0; }
inline int adc1_config_channel_atten(int,int) { return 0; }
