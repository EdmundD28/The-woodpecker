#pragma once
#include "Arduino.h"
#include "Wire.h"
constexpr int SSD1306_WHITE=1,SSD1306_SWITCHCAPVCC=2;
struct Adafruit_SSD1306:TextSink {
    int beginCalls=0;
    std::deque<bool> beginResults;
    Adafruit_SSD1306(int,int,MockWire*,int,uint32_t=400000,uint32_t=100000) {}
    void clearDisplay() { text.clear(); }
    void setTextSize(int) {} void setTextColor(int) {} void setCursor(int,int) {}
    void display() {}
    bool begin(int,int,bool,bool) {
        ++beginCalls;
        if(beginResults.empty()) return true;
        bool result=beginResults.front(); beginResults.pop_front(); return result;
    }
};
