#pragma once
// Woodpecker-themed screens for the 128x64 white SSD1306.
// The bird clings to a trunk on the left and climbs to the hole it will peck
// next: top = first end, middle, bottom = other end. Filled holes are accepted
// taps. Everything here draws only while the ADC is stopped; animations block.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <Adafruit_GFX.h>
#include "fonts/SourceSerif4-Bold10pt7b.h"
#include "fonts/SourceSerif4-Bold9pt7b.h"
#include "fonts/TomThumb.h"
#include "woodpecker_art.h"

namespace oled_ui {
using woodpecker_art::Pose;
constexpr int16_t W=128, H=64;
constexpr int16_t TRUNK_W=12, HOLE_X=6, HOLE_R=3;
constexpr int16_t HOLE_Y[3]={23,33,43};
constexpr int16_t BIRD_X=10, BEAK_DY=10; // beak tip row inside the sprite
constexpr int16_t TEXT_X=46, TEXT_W=W-TEXT_X;
constexpr uint16_t ON=1, OFF=0;

inline int16_t perchY(uint8_t perch) { return HOLE_Y[perch>2?2:perch]-BEAK_DY; }

struct Scene {
    Pose pose=woodpecker_art::AWAKE;
    int16_t birdY=perchY(1);
    int8_t birdDx=0;
    uint8_t holes=0;      // accepted points, drawn as filled peck holes
    int8_t target=-1;     // hole to mark as the next tap, -1 for none
    const char* title="";
    const char* line1="";
    const char* line2="";
    const char* mode="";
};
inline Scene at(uint8_t holes,Pose pose,const char* mode) {
    Scene s; s.pose=pose; s.holes=holes; s.mode=mode;
    s.target=holes<3?int8_t(holes):-1;
    s.birdY=perchY(holes<3?holes:1);
    return s;
}

template<class D> void centred(D& g,const char* text,int16_t baseline,int16_t x0=TEXT_X,int16_t w=TEXT_W) {
    int16_t bx,by; uint16_t bw,bh;
    g.getTextBounds(text,0,baseline,&bx,&by,&bw,&bh);
    g.setCursor(x0+(w-int16_t(bw))/2-bx,baseline);
    g.print(text);
}

template<class D> void trunk(D& g,uint8_t holes,int8_t target) {
    g.drawBitmap(0,0,woodpecker_art::TREE,woodpecker_art::TREE_W,woodpecker_art::TREE_H,ON);
    for(uint8_t i=0;i<3;++i) {
        int16_t cy=HOLE_Y[i];
        g.fillCircle(HOLE_X,cy,HOLE_R+1,OFF);
        if(i<holes) g.fillCircle(HOLE_X,cy,HOLE_R,ON);
        else if(i==target) { g.drawCircle(HOLE_X,cy,HOLE_R,ON); g.drawPixel(HOLE_X,cy,ON); }
        else g.drawCircle(HOLE_X,cy,1,ON);
    }
}

template<class D> void modeTag(D& g,const char* mode) {
    if(!mode || !*mode) return;
    g.setFont(nullptr); g.setTextSize(1);
    int16_t bx,by; uint16_t bw,bh;
    g.getTextBounds(mode,0,0,&bx,&by,&bw,&bh);
    int16_t x=W-int16_t(bw)-4, y=H-10;
    g.fillRoundRect(x-2,y-1,bw+4,10,3,ON);
    g.setTextColor(OFF); g.setCursor(x,y+1); g.print(mode); g.setTextColor(ON);
}

template<class D> void draw(D& g,const Scene& s) {
    g.clearDisplay(); g.setTextWrap(false); g.setTextColor(ON); g.setTextSize(1);
    trunk(g,s.holes,s.target);
    g.drawBitmap(BIRD_X+s.birdDx,s.birdY,woodpecker_art::BIRD_MASKS[s.pose],
                 woodpecker_art::BIRD_W,woodpecker_art::BIRD_H,OFF);
    g.drawBitmap(BIRD_X+s.birdDx,s.birdY,woodpecker_art::BIRD_POSES[s.pose],
                 woodpecker_art::BIRD_W,woodpecker_art::BIRD_H,ON);
    g.setFont(&SourceSerif4_Bold10pt7b); centred(g,s.title,17);
    g.setFont(nullptr);
    centred(g,s.line1,25);
    centred(g,s.line2,35);
    modeTag(g,s.mode);
}

// ---- Decorations -------------------------------------------------------
template<class D> void sparkle(D& g,int16_t x,int16_t y,int16_t r) {
    g.drawFastHLine(x-r,y,2*r+1,ON); g.drawFastVLine(x,y-r,2*r+1,ON);
    if(r>1) { g.drawPixel(x-1,y-1,ON); g.drawPixel(x+1,y-1,ON); g.drawPixel(x-1,y+1,ON); g.drawPixel(x+1,y+1,ON); }
}
template<class D> void zzz(D& g,int16_t x,int16_t y) {
    g.setFont(&TomThumb); g.setCursor(x,y); g.print("z");
    g.setFont(nullptr); g.setCursor(x+4,y-9); g.print("Z");
}
template<class D> void impact(D& g,int16_t y) {
    // Burst lines where the beak meets the bark, plus flying chips.
    int16_t x=TRUNK_W+1;
    g.drawLine(x,y-4,x+3,y-10,ON); g.drawLine(x+2,y-4,x+7,y-9,ON);
    g.drawLine(x,y+4,x+3,y+10,ON); g.drawLine(x+2,y+4,x+7,y+9,ON);
}
template<class D> void chips(D& g,int16_t y,uint8_t step) {
    static const int8_t dx[]={1,3,2,4}, dy[]={-4,-6,4,6};
    for(uint8_t i=0;i<4;++i) g.fillRect(TRUNK_W+1+dx[i]*step/2,y+dy[i]*step/2,2,2,ON);
}
template<class D> void sweat(D& g,int16_t x,int16_t y) {
    g.drawPixel(x,y,ON); g.drawFastHLine(x-1,y+1,3,ON); g.fillCircle(x,y+3,1,ON);
}

inline const char* pointName(uint8_t accepted) {
    return accepted==0?"first end":accepted==1?"middle":"other end";
}

// ---- Static screens ----------------------------------------------------
template<class D> void splash(D& g,const char* mode) {
    Scene s=at(0,woodpecker_art::HAPPY,mode); s.title="Hi!"; s.line1="I'm Woody"; s.line2="waking up...";
    draw(g,s); sparkle(g,BIRD_X+32,s.birdY+2,2); sparkle(g,TEXT_X+14,48,1); g.display();
}
template<class D> void preparing(D& g,uint8_t holes,const char* mode) {
    Scene s=at(holes,woodpecker_art::AWAKE,mode); s.title="..."; s.line1="Preparing"; s.line2="one sec";
    draw(g,s); g.display();
}
template<class D> void locked(D& g,uint8_t holes,const char* mode) {
    Scene s=at(holes,woodpecker_art::SLEEP,mode); s.title="HOLD"; s.line1="the touch pad"; s.line2="to wake me";
    draw(g,s); zzz(g,BIRD_X+29,s.birdY+4); g.display();
}
template<class D> void arming(D& g,uint8_t holes,const char* mode) {
    Scene s=at(holes,woodpecker_art::AWAKE,mode); s.title="HOLD..."; s.line1="keep holding"; s.line2=pointName(holes);
    draw(g,s); g.display();
}
template<class D> void ready(D& g,uint8_t holes,const char* mode) {
    Scene s=at(holes,woodpecker_art::ALERT,mode); s.title="TAP!"; s.line1=pointName(holes); s.line2="keep holding";
    draw(g,s);
    // Surprise mark next to the crest.
    int16_t x=BIRD_X+32, y=s.birdY+2;
    g.drawFastVLine(x,y,5,ON); g.drawFastVLine(x+1,y,5,ON); g.drawFastHLine(x,y+7,2,ON);
    g.display();
}
template<class D> void releaseFirst(D& g,uint8_t holes,const char* mode) {
    Scene s=at(holes,woodpecker_art::AWAKE,mode); s.title="LET GO"; s.line1="then hold"; s.line2="again";
    draw(g,s); g.display();
}
template<class D> void cancelled(D& g,uint8_t holes,const char* mode) {
    Scene s=at(holes,woodpecker_art::DIZZY,mode); s.title="OOPS"; s.line1="let go early"; s.line2="hold to retry";
    draw(g,s); g.display();
}
template<class D> void groupReset(D& g,const char* mode) {
    Scene s=at(0,woodpecker_art::AWAKE,mode); s.title="RESET"; s.line1="start at"; s.line2="an end";
    draw(g,s); g.display();
}
// Fatal errors: the code stays visible in tiny type for whoever debugs it.
inline const char* errorHint(const char* reason) {
    if(!strncmp(reason,"ADC_",4)) return "sensor issue";
    if(!strncmp(reason,"MODEL_",6)) return "memory issue";
    return "glitch";
}
template<class D> void error(D& g,const char* reason) {
    Scene s=at(0,woodpecker_art::DIZZY,""); s.title="ERROR"; s.line1=""; s.line2="";
    draw(g,s);
    g.setFont(&TomThumb); centred(g,reason,25); g.setFont(nullptr);
    centred(g,errorHint(reason),31); centred(g,"restart me",41);
    g.display();
}
// Model could not score the group: nothing to show, start a new group.
template<class D> void modelError(D& g) {
    Scene s=at(3,woodpecker_art::WORRIED,""); s.title="HMM?"; s.line1="can't decide"; s.line2="hold to retry";
    draw(g,s); sweat(g,BIRD_X+31,s.birdY+5); g.display();
}
template<class D> void thinking(D& g,const char* mode) {
    Scene s=at(3,woodpecker_art::AWAKE,mode); s.title="HMM..."; s.line1="listening to"; s.line2="the wood";
    draw(g,s);
    g.drawCircle(BIRD_X+31,s.birdY+3,1,ON); g.drawCircle(BIRD_X+34,s.birdY-1,2,ON);
    g.display();
}

// ---- Animations (blocking; ADC must already be stopped) -----------------
// One step is one full frame: about 25 ms of I2C at 400 kHz plus the wait.
// One look per rejection reason from the firmware's quality check.
struct Problem { const char* line1; const char* line2; Pose pose; };
inline Problem problemFor(const char* reason) {
    if(!strcmp(reason,"CLIPPED")) return {"too hard!","tap softer",woodpecker_art::DIZZY};
    if(!strcmp(reason,"WEAK_SIGNAL")) return {"too soft","tap harder",woodpecker_art::WORRIED};
    if(!strcmp(reason,"SAMPLE_LOSS")) return {"missed it","tap again",woodpecker_art::WORRIED};
    if(!strcmp(reason,"PREPROCESS_ERROR")) return {"can't read it","tap again",woodpecker_art::DIZZY};
    return {reason,"hold & retry",woodpecker_art::DIZZY};
}
template<class D,class Wait> void invalid(D& g,Wait wait,uint8_t holes,const char* reason,const char* mode) {
    static const int8_t shake[]={-2,2,-2,2,0};
    const Problem p=problemFor(reason);
    Scene s=at(holes,p.pose,mode); s.title="OOPS!"; s.line1=p.line1; s.line2=p.line2;
    for(int8_t dx:shake) {
        s.birdDx=dx; draw(g,s);
        if(p.pose==woodpecker_art::WORRIED) sweat(g,BIRD_X+31+dx,s.birdY+5);
        g.display(); wait(60);
    }
}

// About 1 s: wind up, two pecks (the hole appears on the second), a happy
// hop, then the bird climbs to the next hole.
template<class D,class Wait> void accepted(D& g,Wait wait,uint8_t holesAfter,const char* mode) {
    uint8_t point=holesAfter-1;
    Scene s=at(point,woodpecker_art::PECK1,mode);
    int16_t hole=HOLE_Y[point];
    for(uint8_t peck=0;peck<2;++peck) {
        s.pose=woodpecker_art::AWAKE; s.birdDx=1; s.title=""; draw(g,s); g.display(); wait(60);
        if(peck==1) s.holes=holesAfter;
        s.pose=woodpecker_art::PECK2; s.birdDx=-1; s.title="TOK!"; draw(g,s);
        impact(g,hole); g.display(); wait(90);
        s.pose=woodpecker_art::PECK1; s.birdDx=0; draw(g,s); chips(g,hole,3); g.display(); wait(50);
    }
    s.pose=woodpecker_art::HAPPY; s.target=-1; s.title="NICE!";
    char count[12]; snprintf(count,sizeof(count),"%u/3 done",unsigned(holesAfter));
    s.line1=count; s.line2=holesAfter<3?pointName(holesAfter):"all done!";
    int16_t base=s.birdY;
    for(uint8_t hop=0;hop<2;++hop) {
        s.birdY=base-3; draw(g,s); sparkle(g,BIRD_X+32,s.birdY+2,2); g.display(); wait(90);
        s.birdY=base; draw(g,s); sparkle(g,BIRD_X+34,s.birdY+4,1); g.display(); wait(90);
    }
    if(holesAfter<3) {
        int16_t goal=perchY(holesAfter);
        s.target=int8_t(holesAfter); s.pose=woodpecker_art::AWAKE;
        for(uint8_t step=1;step<=3;++step) {
            s.birdY=base+(goal-base)*step/3; draw(g,s); g.display(); wait(50);
        }
    }
}

// Wood 1 = solid and sound. Wood 2 = hollow, hidden defect. The class number
// stays on the serial port for the back end; the audience sees the verdict.
template<class D> void warningSign(D& g,int16_t x,int16_t y,bool filled) {
    if(filled) g.fillTriangle(x,y-7,x-8,y+6,x+8,y+6,ON);
    else g.drawTriangle(x,y-7,x-8,y+6,x+8,y+6,ON);
    uint16_t c=filled?OFF:ON;
    g.drawFastVLine(x,y-2,5,c); g.drawPixel(x,y+4,c);
}
} // namespace oled_ui
