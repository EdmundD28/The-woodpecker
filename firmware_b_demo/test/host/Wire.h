#pragma once
struct MockWire {
    void begin(int,int) {} void setTimeOut(int) {}
    void beginTransmission(int) {} int endTransmission() { return 0; }
};
inline MockWire Wire;
