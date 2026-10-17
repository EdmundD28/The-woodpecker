// Execute actual firmware with simulated Arduino/I2C/DMA interfaces.
#include "../../src/main.cpp"
#include "reference_vectors.h"
#include <cassert>
#include <iostream>

void reset() {
    fakeNow=1000; head=filled=0; next=PRE; discard=lossHistory=0;
    active=loss=requested=baselineReady=finishedBlock=false;
    running=readyShown=true; rearmAt=resumeAt=0; baseline=0; accepted=0;
    fakeAdc.clear(); fakeEvents.clear(); Serial.text.clear(); Serial.commands.clear(); oled.text.clear();
}
void waveform(int amplitude) {
    for(size_t i=0;i<N;++i) raw[i]=2048+(i%2?amplitude:-amplitude);
}
void readyAgain() {
    fakeNow=resumeAt; restartAdc();
    fakeNow=rearmAt; readyShown=true; discard=0;
}
void captureReference(size_t row,size_t point) {
    std::copy(test_raw[row][point],test_raw[row][point]+N,raw);
    finishCapture();
}
int main() {
    reset(); setup();
    assert(N==2400 && FS==8000 && PRE==256 && THRESHOLD==500);
    assert(oled.text.find("Preparing")!=std::string::npos && !readyShown);
    assert(OLED_SDA==21 && OLED_SCL==22 && OLED_ADDRESS==0x3C);
    std::cout<<"PASS initialization, trained sampling contract and OLED pins\n";

    float maxInputError=0,maxScoreError=0;
    for(size_t row=0;row<vector_count;++row) {
        for(size_t p=0;p<3;++p) assert(wood_classifier::store_point(test_raw[row][p],p));
        for(size_t i=0;i<wood_classifier::point_length*3;++i)
            maxInputError=std::max(maxInputError,std::fabs(wood_classifier::input[i]-expected_input[row][i]));
        float scores[2]; assert(wood_classifier::classify(scores));
        for(size_t c=0;c<2;++c) maxScoreError=std::max(maxScoreError,std::fabs(scores[c]-expected_scores[row][c]));
        assert((scores[1]>scores[0])==(expected_scores[row][1]>expected_scores[row][0]));
        for(size_t p=0;p<3;++p) assert(wood_classifier::store_point(test_raw[row][2-p],p));
        float reversed[2]; assert(wood_classifier::classify(reversed));
        assert(std::fabs(reversed[0]-scores[0])<1e-7f && std::fabs(reversed[1]-scores[1])<1e-7f);
    }
    assert(maxInputError<=1e-7f && maxScoreError<=1e-4f);
    assert(!wood_classifier::store_point(test_raw[0][0],3));
    std::cout<<"PASS "<<vector_count<<" real triplets and both directions; max_input_error="
             <<maxInputError<<", max_score_error="<<maxScoreError<<"\n";

    for(size_t row=0;row<vector_count;++row) {
        reset();
        for(size_t p=0;p<3;++p) {
            captureReference(row,p); assert(accepted==p+1 && !running);
            if(p<2) {
                assert(Serial.text.find("#RESULT")==std::string::npos);
                assert(oled.text.find("WOOD:")==std::string::npos); readyAgain();
            }
        }
        int winner=expected_scores[row][1]>expected_scores[row][0]?2:1;
        assert(oled.text.find("WOOD: "+std::to_string(winner))!=std::string::npos);
        assert(Serial.text.find("#RESULT,wood="+std::to_string(winner))!=std::string::npos);
        fakeNow=resumeAt-1; loop(); assert(accepted==3 && !running);
        ++fakeNow; loop(); assert(accepted==0 && running && !readyShown);
    }
    std::cout<<"PASS triplet flow, no per-hit classification, five-second reset\n";

    for(int kind=0;kind<3;++kind) {
        reset(); captureReference(0,0); readyAgain();
        float saved=wood_classifier::input[0]; waveform(600);
        if(kind==0) raw[0]=4095;
        if(kind==1) loss=true;
        if(kind==2) waveform(10);
        finishCapture(); assert(accepted==1 && wood_classifier::input[0]==saved);
        const char* expected=kind==0?"CLIPPED":kind==1?"SAMPLE_LOSS":"WEAK_SIGNAL";
        assert(oled.text.find(expected)!=std::string::npos);
        readyAgain(); captureReference(0,1); readyAgain(); captureReference(0,2);
        assert(accepted==3 && Serial.text.find("#RESULT,wood=1")!=std::string::npos);
    }
    std::cout<<"PASS invalid hit keeps position/earlier wave; retry completes correct triplet\n";

    reset();
    for(size_t i=0;i<PRE+20;++i) sample(1800+i%PRE);
    size_t oldest=head;
    sample(3048); assert(active && next==PRE+1 && raw[PRE]==3048);
    for(size_t i=0;i<PRE;++i) assert(raw[i]==ring[(oldest+i)%PRE]);
    for(size_t i=PRE+1;i<N;++i) sample(i%2?2648:1448);
    assert(!active && accepted==1 && next==N);
    std::cout<<"PASS hardware threshold, ordered prebuffer, exactly 2400 points\n";

    reset(); for(size_t i=0;i<PRE;++i) sample(2048);
    requested=true; sample(2048); assert(active);
    waveform(0); loss=false; finishCapture(); assert(accepted==0);
    reset(); active=true; next=PRE+1; captureDeadline=fakeNow-1; loop();
    assert(!active && !running && accepted==0 && oled.text.find("SAMPLE_LOSS")!=std::string::npos);
    std::cout<<"PASS software trigger, weak-signal rejection and incomplete capture timeout\n";

    reset(); active=true; next=PRE; captureDeadline=fakeNow+600;
    fakeEvents.push_back(I2S_EVENT_DMA_ERROR); checkEvents(); assert(loss && lossHistory==PRE);
    reset(); readyShown=false; rearmAt=fakeNow+500;
    for(size_t i=0;i<PRE;++i) sample(2048);
    sample(3000); assert(!active);
    fakeNow=rearmAt; loop(); assert(readyShown && oled.text.find("first endpoint")!=std::string::npos);
    reset(); markLoss(); assert(lossHistory==PRE);
    for(size_t i=0;i<PRE;++i) sample(2048);
    assert(lossHistory==0);
    fakeNow=0xFFFFFFF0u; assert(!due(10)); fakeNow=11; assert(due(10));
    std::cout<<"PASS DMA loss, preparing/rearming gate, recovery and timer wraparound\n";

    reset(); accepted=2; Serial.commands.push_back('R'); loop();
    assert(accepted==0 && !running && !requested && Serial.text.find("#RESET")!=std::string::npos);
    fakeNow=resumeAt; loop(); assert(running && !readyShown);
    std::cout<<"PASS manual group reset\nALL PASSED ("<<wood_classifier::mode<<")\n";
}
