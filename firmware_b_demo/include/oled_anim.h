#pragma once
// Boot, result and hole-patching animations, built on oled_ui.h.
// Like everything in oled_ui, these block and must run with the ADC stopped.
#include "oled_ui.h"
#include "fonts/SourceSerif4-Bold12pt7b.h"
#include "fly_art.h"

namespace oled_ui {
using namespace woodpecker_art;

template<class D> void birdAt(D& g,Pose p,int16_t x,int16_t y) {
    g.drawBitmap(x,y,BIRD_MASKS[p],BIRD_W,BIRD_H,OFF);
    g.drawBitmap(x,y,BIRD_POSES[p],BIRD_W,BIRD_H,ON);
}
template<class D> void flyAt(D& g,const uint8_t* bmp,const uint8_t* mask,int16_t x,int16_t y) {
    g.drawBitmap(x,y,mask,FLY_W,FLY_H,OFF);
    g.drawBitmap(x,y,bmp,FLY_W,FLY_H,ON);
}
// Tree with only the rows from `from` downward visible (grows upward).
template<class D> void treeGrow(D& g,int16_t from,uint8_t dots) {
    g.drawBitmap(0,0,TREE,TREE_W,TREE_H,ON);
    if(from>0) g.fillRect(0,0,TREE_W,from,OFF);
    for(uint8_t i=0;i<dots && i<3;++i) {
        int16_t cy=HOLE_Y[i];
        g.fillCircle(HOLE_X,cy,HOLE_R+1,OFF); g.drawCircle(HOLE_X,cy,1,ON);
    }
}
template<class D> void logo(D& g,uint8_t letters,int16_t dy=0) {
    static const char* name="Woody";
    char buf[6]={0}; for(uint8_t i=0;i<letters && i<5;++i) buf[i]=name[i];
    g.setFont(&SourceSerif4_Bold12pt7b);
    // Centre on the full word so letters do not shift while typing.
    int16_t bx,by; uint16_t bw,bh; g.getTextBounds(name,0,0,&bx,&by,&bw,&bh);
    g.setCursor(TEXT_X+(TEXT_W-int16_t(bw))/2-bx,20+dy); g.print(buf);
    g.setFont(nullptr);
}

// ---- Boot: tree grows, bird climbs in, name types out, two knocks --------
template<class D,class Wait> void boot(D& g,Wait wait,const char* mode) {
    g.setTextWrap(false); g.setTextColor(ON);
    // 1. Trunk grows up from the root, canopy last.
    for(int16_t from=64;from>=0;from-=8) { g.clearDisplay(); treeGrow(g,from,0); g.display(); wait(45); }
    // 2. The three peck marks appear.
    for(uint8_t d=1;d<=3;++d) { g.clearDisplay(); treeGrow(g,0,d); g.display(); wait(70); }
    // 3. Bird flies in from the top right in bounding flight and lands on the trunk.
    const int16_t goal=perchY(1);
    struct Step { const uint8_t* b; const uint8_t* m; int16_t x,y; uint16_t ms; };
    const Step path[]={
        {FLY_UP,MASK_FLY_UP,     118,-26,60},
        {FLY_MID,MASK_FLY_MID,   104,-18,60},
        {FLY_DOWN,MASK_FLY_DOWN,  91,-13,60},
        {FLY_FOLD,MASK_FLY_FOLD,  78, -6,70},
        {FLY_FOLD,MASK_FLY_FOLD,  66,  0,70},
        {FLY_UP,MASK_FLY_UP,      55,  2,60},
        {FLY_DOWN,MASK_FLY_DOWN,  44,  6,60},
        {FLY_MID,MASK_FLY_MID,    35, 10,60},
        {FLY_LAND,MASK_FLY_LAND,  24, 13,80},
        {FLY_LAND,MASK_FLY_LAND,  17, 16,90},
    };
    for(const Step& st:path) { g.clearDisplay(); treeGrow(g,0,3); flyAt(g,st.b,st.m,st.x,st.y); g.display(); wait(st.ms); }
    g.clearDisplay(); treeGrow(g,0,3); birdAt(g,AWAKE,BIRD_X+2,goal-2); g.display(); wait(70);
    g.clearDisplay(); treeGrow(g,0,3); birdAt(g,AWAKE,BIRD_X,goal); g.display(); wait(200);
    // 4. Blink, then look up alert.
    g.clearDisplay(); treeGrow(g,0,3); birdAt(g,SLEEP,BIRD_X,goal); g.display(); wait(80);
    g.clearDisplay(); treeGrow(g,0,3); birdAt(g,ALERT,BIRD_X,goal); g.display(); wait(160);
    // 5. Name types out, a sparkle on each letter.
    for(uint8_t n=1;n<=5;++n) {
        g.clearDisplay(); treeGrow(g,0,3); birdAt(g,ALERT,BIRD_X,goal); logo(g,n);
        sparkle(g,TEXT_X+6+n*15,3+(n%2)*2,n%2?2:1);
        g.display(); wait(n<5?80:140);
    }
    // 6. Two knocks; each one prints a word.
    static const char* words[]={"knock","knock knock!"};
    for(uint8_t k=0;k<2;++k) {
        g.clearDisplay(); treeGrow(g,0,3); birdAt(g,AWAKE,BIRD_X+1,goal); logo(g,5);
        if(k) { g.setFont(nullptr); g.setCursor(TEXT_X+(TEXT_W-72)/2,32); g.print("knock"); }
        g.display(); wait(90);
        g.clearDisplay(); treeGrow(g,0,3); birdAt(g,PECK2,BIRD_X-1,goal); logo(g,5);
        impact(g,HOLE_Y[1]);
        g.setFont(nullptr); g.setCursor(TEXT_X+(TEXT_W-72)/2,32); g.print(words[k]);
        g.display(); wait(110);
        g.clearDisplay(); treeGrow(g,0,3); birdAt(g,PECK1,BIRD_X,goal); logo(g,5); chips(g,HOLE_Y[1],3);
        g.setCursor(TEXT_X+(TEXT_W-72)/2,32); g.print(words[k]);
        g.display(); wait(70);
    }
    // 7. Happy hop, mode tag, hold.
    for(uint8_t hop=0;hop<3;++hop) {
        int16_t y=goal-(hop%2?0:3);
        g.clearDisplay(); treeGrow(g,0,3); birdAt(g,HAPPY,BIRD_X,y); logo(g,5);
        g.setCursor(TEXT_X+(TEXT_W-72)/2,32); g.print("knock knock!");
        g.setFont(&TomThumb); centred(g,"wood tap tester",48); g.setFont(nullptr);
        sparkle(g,BIRD_X+32,y+2,hop%2?1:2);
        if(hop==2) modeTag(g,mode);
        g.display(); wait(hop<2?100:1400);
    }
}

// ---- Result pieces: verdict badge and healing holes ---------------------
// Specimen 1: filled badge. Specimen 2: outline badge; no defect claim.
template<class D> void badge(D& g,bool solid,int16_t dy) {
    if(solid) g.fillRoundRect(TEXT_X-6,1+dy,TEXT_W+5,22,5,ON);
    else { g.fillRoundRect(TEXT_X-6,1+dy,TEXT_W+5,22,5,OFF); g.drawRoundRect(TEXT_X-6,1+dy,TEXT_W+5,22,5,ON); }
    g.setFont(&SourceSerif4_Bold9pt7b); g.setTextColor(solid?OFF:ON);
    centred(g,solid?"WOOD 1":"WOOD 2",18+dy,TEXT_X-6,TEXT_W+5);
    g.setFont(nullptr); g.setTextColor(ON);
}
template<class D> void pill(D& g,bool solid,const char* t,int16_t y) {
    if(solid) { g.fillRoundRect(TEXT_X+6,y,TEXT_W-12,12,4,ON); g.setTextColor(OFF); }
    else g.drawRoundRect(TEXT_X+6,y,TEXT_W-12,12,4,ON);
    centred(g,t,y+2); g.setTextColor(ON);
}
template<class D> void verdictText(D& g,bool solid,int16_t dy) {
    badge(g,solid,dy);
    if(solid) centred(g,"first wood",34+dy);
    else { centred(g,"second wood",29+dy); centred(g,"identified",39+dy); }
}
// holeR[i]: radius of each filled hole (3 = full, 0 = healed to a dot).
template<class D> void trunkHeal(D& g,const uint8_t* holeR,int8_t target) {
    g.drawBitmap(0,0,TREE,TREE_W,TREE_H,ON);
    for(uint8_t i=0;i<3;++i) {
        int16_t cy=HOLE_Y[i];
        g.fillCircle(HOLE_X,cy,HOLE_R+1,OFF);
        if(i==target && holeR[i]==0) { g.drawCircle(HOLE_X,cy,HOLE_R,ON); g.drawPixel(HOLE_X,cy,ON); }
        else if(holeR[i]>=2) g.fillCircle(HOLE_X,cy,holeR[i],ON);
        else g.drawCircle(HOLE_X,cy,1,ON);
    }
}
template<class D> void endFrame(D& g,const uint8_t* r,int8_t target,Pose p,int16_t by,int16_t bdx=0) {
    g.clearDisplay(); g.setTextWrap(false); g.setTextColor(ON);
    trunkHeal(g,r,target); birdAt(g,p,BIRD_X+bdx,by);
}
// ---- Result state (stays until the pad is held) --------------------------
// Same look as verdictScreen; in idle the bottom decoration takes turns with
// a hint so people know how to move on.
template<class D> void resultIdle(D& g,bool solid,uint8_t phase) {
    Scene s=at(3,solid?HAPPY:WORRIED,"");
    bool hint=phase>=2;
    if(solid && phase==1) s.birdY-=2;
    draw(g,s);
    badge(g,solid,0);
    if(solid) centred(g,"first wood",34);
    else { centred(g,"second wood",29); centred(g,"identified",39); }
    if(hint) {
        centred(g,"hold for next",52);
        if(phase==3) { g.drawFastHLine(TEXT_X+4,61,TEXT_W-8,ON); }   // underline pulse
    } else if(solid) {
        const int16_t sx[]={TEXT_X+12,TEXT_X+70,TEXT_X+24,TEXT_X+58}, sy[]={44,42,56,54};
        for(uint8_t i=0;i<4;++i) sparkle(g,sx[i],sy[i],((i+phase)%2)?2:1);
        sparkle(g,BIRD_X+32,s.birdY+2,phase?2:1);
    } else {
        warningSign(g,TEXT_X+TEXT_W/2,56,phase==0);
        sweat(g,BIRD_X+31,s.birdY+5);
    }
    g.display();
}
template<class D> void resultIntro(D& g,bool solid,uint8_t step) {
    Scene s=at(3,solid?HAPPY:WORRIED,"");
    if(solid && step%2==1) s.birdY-=3;
    if(!solid && step<4) s.birdDx=step%2?1:-1;
    draw(g,s); badge(g,solid,0);
    if(solid) {
        centred(g,"first wood",34);
        const int16_t sx[]={TEXT_X+12,TEXT_X+70,TEXT_X+24,TEXT_X+58}, sy[]={44,42,56,54};
        for(uint8_t i=0;i<4;++i) sparkle(g,sx[i],sy[i],((i+step)%2)?2:1);
        sparkle(g,BIRD_X+32,s.birdY+2,step%2?2:1);
    } else {
        centred(g,"second wood",29); centred(g,"identified",39);
        warningSign(g,TEXT_X+TEXT_W/2,56,step%2==0);
        sweat(g,BIRD_X+31+s.birdDx,s.birdY+5);
    }
    g.display();
}
template<class D,class Wait> void result(D& g,Wait wait,bool solid,uint8_t cycles) {
    for(uint8_t i=0;i<6;++i) { resultIntro(g,solid,i); wait(110); }
    for(uint8_t c=0;c<cycles;++c) {
        resultIdle(g,solid,0); wait(500); resultIdle(g,solid,1); wait(500);
        resultIdle(g,solid,2); wait(700); resultIdle(g,solid,3); wait(700);
    }
}

// ---- Patch state (entered when the pad is held on the result) ------------
template<class D,class Wait> void patch(D& g,Wait wait,bool solid) {
    uint8_t r[3]={3,3,3};
    const int16_t mid=perchY(1);
    // Hold registered: bird perks up, hint confirms.
    endFrame(g,r,-1,ALERT,mid); verdictText(g,solid,0);
    g.fillRect(TEXT_X-6,46,TEXT_W+6,18,OFF);
    pill(g,solid,"OK, next!",solid?48:50);
    g.display(); wait(350);
    for(int16_t dy=-8;dy>-64;dy-=12) { endFrame(g,r,-1,AWAKE,mid); verdictText(g,solid,dy); g.display(); wait(45); }
    // 2. "DONE!" line while the bird nods.
    for(uint8_t i=0;i<2;++i) {
        endFrame(g,r,-1,i?HAPPY:AWAKE,mid);
        g.setFont(&SourceSerif4_Bold10pt7b); centred(g,"DONE!",17); g.setFont(nullptr);
        centred(g,"patching",25); centred(g,"the holes...",35);
        g.display(); wait(i?250:150);
    }
    // 3. Bird hops down, healing holes bottom to top as it passes.
    static const int16_t perch[]={perchY(0),perchY(1),perchY(2)};
    for(int8_t h=2;h>=0;--h) {
        endFrame(g,r,-1,AWAKE,perch[h]);
        g.setFont(&SourceSerif4_Bold10pt7b); centred(g,"DONE!",17); g.setFont(nullptr);
        centred(g,"patching",25); centred(g,"the holes...",35);
        g.display(); wait(90);
        for(uint8_t s=2;s<=3;++s) {
            r[h]=s==2?2:0;
            endFrame(g,r,-1,s==2?PECK1:AWAKE,perch[h],s==2?-1:0);
            g.setFont(&SourceSerif4_Bold10pt7b); centred(g,"DONE!",17); g.setFont(nullptr);
            centred(g,"patching",25); centred(g,"the holes...",35);
            sparkle(g,HOLE_X+7,HOLE_Y[h]-3,s==2?1:2);
            g.display(); wait(80);
        }
    }
    // 4. Back at the first hole: ready for the next piece.
    for(uint8_t i=0;i<3;++i) {
        int16_t y=perch[0]-(i==1?3:0);
        endFrame(g,r,0,i==2?ALERT:HAPPY,y);
        g.setFont(&SourceSerif4_Bold10pt7b); centred(g,"NEXT!",17); g.setFont(nullptr);
        centred(g,"fresh wood,",25); centred(g,"please",35);
        if(i<2) sparkle(g,BIRD_X+32,y+2,i?1:2);
        g.display(); wait(i<2?100:1400);
    }
}
} // namespace oled_ui
