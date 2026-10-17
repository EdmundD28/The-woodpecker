#pragma once
#include "Arduino.h"
struct MockWire {
    int clock=0,ends=0;
    std::deque<int> probeResults;
    void begin(int,int) {} void setTimeOut(int) {}
    void setClock(int value) { clock=value; }
    void end() { ++ends; }
    void beginTransmission(int) {}
    int endTransmission() {
        if(probeResults.empty()) return 0;
        int result=probeResults.front(); probeResults.pop_front(); return result;
    }
};
inline MockWire Wire;
