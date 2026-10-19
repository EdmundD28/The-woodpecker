#pragma once
#include "Arduino.h"
#include "Wire.h"
#include "Adafruit_GFX.h"
constexpr int SSD1306_WHITE=1,SSD1306_SWITCHCAPVCC=2;
// Records the text of the current frame (one line per print) and counts frames.
// Drawing calls are accepted and ignored; the woodpecker art is checked by the
// preview renders, not here.
struct Adafruit_SSD1306:TextSink {
    int beginCalls=0, frames=0;
    std::string displayedText;
    std::deque<bool> beginResults;
    Adafruit_SSD1306(int,int,MockWire*,int,uint32_t=400000,uint32_t=100000) {}
    void clearDisplay() { text.clear(); }
    void setTextSize(int) {} void setTextColor(int) {} void setCursor(int,int) {}
    void setTextWrap(bool) {} void setFont(const GFXfont*) {}
    void print(const char* s) { text+=s; text+='\n'; }
    void getTextBounds(const char* s,int16_t x,int16_t y,int16_t* bx,int16_t* by,uint16_t* bw,uint16_t* bh) {
        *bx=x; *by=y-7; *bw=uint16_t(6*std::strlen(s)); *bh=8;
    }
    void drawBitmap(int16_t,int16_t,const uint8_t*,int16_t,int16_t,uint16_t) {}
    void drawPixel(int16_t,int16_t,uint16_t) {}
    void drawLine(int16_t,int16_t,int16_t,int16_t,uint16_t) {}
    void drawFastHLine(int16_t,int16_t,int16_t,uint16_t) {}
    void drawFastVLine(int16_t,int16_t,int16_t,uint16_t) {}
    void fillRect(int16_t,int16_t,int16_t,int16_t,uint16_t) {}
    void drawCircle(int16_t,int16_t,int16_t,uint16_t) {}
    void fillCircle(int16_t,int16_t,int16_t,uint16_t) {}
    void drawRoundRect(int16_t,int16_t,int16_t,int16_t,int16_t,uint16_t) {}
    void fillRoundRect(int16_t,int16_t,int16_t,int16_t,int16_t,uint16_t) {}
    void drawTriangle(int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,uint16_t) {}
    void fillTriangle(int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,uint16_t) {}
    void display() { ++frames; displayedText+=text; }
    bool begin(int,int,bool,bool) {
        ++beginCalls;
        if(beginResults.empty()) return true;
        bool result=beginResults.front(); beginResults.pop_front(); return result;
    }
};
