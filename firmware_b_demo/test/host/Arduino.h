#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <deque>
#include <algorithm>
#include <cstring>
inline uint32_t fakeNow=0;
inline uint32_t millis() { return fakeNow; }
inline void delay(uint32_t ms) { fakeNow+=ms; }
constexpr int LOW=0,HIGH=1,INPUT_PULLDOWN=2,INPUT_PULLUP=3,OUTPUT=4;
inline int fakeOutputs[40]={};
inline void digitalWrite(int pin,int value) { fakeOutputs[pin]=value; }
inline int fakeTouchLevel=LOW,fakeInputPin=-1,fakeInputMode=-1;
inline void pinMode(int pin,int mode) { fakeInputPin=pin; fakeInputMode=mode; }
inline int digitalRead(int pin) { return pin==27?fakeTouchLevel:LOW; }
struct TextSink {
    std::string text;
    std::deque<char> commands;
    void begin(int) {}
    int available() { return int(commands.size()); }
    char read() { char c=commands.front(); commands.pop_front(); return c; }
    void println(const char* s) { text+=s; text+='\n'; }
    template<class... A> void printf(const char* format,A... args) {
        char buf[512]; std::snprintf(buf,sizeof(buf),format,args...); text+=buf;
    }
};
inline TextSink Serial;
