#pragma once
#include "Arduino.h"
#include "Wire.h"
constexpr int SSD1306_WHITE=1,SSD1306_SWITCHCAPVCC=2;
struct Adafruit_SSD1306:TextSink {
    Adafruit_SSD1306(int,int,MockWire*,int) {}
    void clearDisplay() { text.clear(); }
    void setTextSize(int) {} void setTextColor(int) {} void setCursor(int,int) {}
    void display() {} bool begin(int,int,bool,bool) { return true; }
};
