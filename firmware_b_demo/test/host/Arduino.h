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
