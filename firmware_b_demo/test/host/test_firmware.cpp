// Run the ACTUAL firmware with simulated Arduino/I2C/DMA surfaces.
#include "../../src/main.cpp"
#include <cassert>
#include <iostream>

void reset() {
    fakeNow=1000; head=filled=0; next=PRE; discard=lossHistory=0;
    active=loss=requested=baselineReady=finishedBlock=false;
    running=readyShown=true; rearmAt=0; resumeAt=0; baseline=0;
    accepted=0; for(auto& v:votes) v=0;
    fakeAdc.clear(); fakeEvents.clear(); Serial.text.clear(); oled.text.clear();
}
void waveform(int amplitude) {
    // Symmetric synthetic signal with exactly mean 2048.
    for(size_t i=0;i<N;++i) raw[i]=2048+(i%2?amplitude:-amplitude);
}
void readyAgain() {
    fakeNow=resumeAt; restartAdc();
    fakeNow=rearmAt; readyShown=true; discard=0;
}
int main() {
    reset(); setup();
    assert(oled.text.find("Preparing")!=std::string::npos && !readyShown);
    assert(OLED_SDA==21 && OLED_SCL==22 && OLED_ADDRESS==0x3C);
    std::cout<<"PASS initialization and OLED pin constants\n";

    for(int i=0;i<4;++i) {
        reset(); waveform(200+i*400);
        assert(!quality() && preprocess());
        float sum=0; for(float v:modelInput) sum+=v;
        assert(fabsf(sum)<0.001f);
        assert(fabsf(modelInput[1]-float(200+i*400)/2048)<1e-6);
        assert(classifyPlaceholder(modelInput)==i);
        std::cout<<"PASS synthetic amplitude="<<200+i*400<<" -> DEMO "<<i+1<<"\n";
    }
    reset(); waveform(600); finishCapture();
    assert(accepted==1 && votes[1]==1 && !running);
    assert(oled.text.find("1/3")!=std::string::npos && oled.text.find("DEMO: 2")!=std::string::npos);
    readyAgain(); waveform(600); finishCapture();
    readyAgain(); waveform(1000); finishCapture();
    assert(accepted==3 && oled.text.find("Final DEMO result")!=std::string::npos);
    assert(oled.text.find("DEMO: 2")!=std::string::npos);
    fakeNow=resumeAt-1; loop(); assert(accepted==3 && !running);
    ++fakeNow; loop(); assert(accepted==0 && running && !readyShown);
    std::cout<<"PASS per-tap display, majority 2/2/3 -> 2, five-second reset\n";

    reset();
    for(int amplitude:{200,600,1000}) { waveform(amplitude); finishCapture(); if(accepted<3) readyAgain(); }
    assert(oled.text.find("Uncertain")!=std::string::npos);
    std::cout<<"PASS 1/2/3 -> no majority\n";

    for(int kind=0;kind<3;++kind) {
        reset(); accepted=1; votes[1]=1; waveform(600);
        if(kind==0) raw[0]=4095;
        if(kind==1) loss=true;
        if(kind==2) waveform(10);
        finishCapture(); assert(accepted==1 && votes[1]==1);
        const char* expected=kind==0?"CLIPPED":kind==1?"SAMPLE_LOSS":"WEAK_SIGNAL";
        assert(oled.text.find(expected)!=std::string::npos);
        std::cout<<"PASS rejection: "<<expected<<", counter unchanged\n";
    }
    reset();
    for(size_t i=0;i<PRE+20;++i) sample(1800+i%PRE);
    size_t oldest=head;
    sample(2648); assert(active && next==PRE+1 && raw[PRE]==2648);
    for(size_t i=0;i<PRE;++i) assert(raw[i]==ring[(oldest+i)%PRE]);
    for(size_t i=PRE+1;i<N;++i) sample(i%2?2648:1448);
    assert(!active && accepted==1 && next==N);
    std::cout<<"PASS threshold trigger, ordered 256 prepoints, exactly 1024 capture\n";
    reset(); for(size_t i=0;i<PRE;++i) sample(2048);
    requested=true; sample(2048); assert(active);
    waveform(0); loss=false; finishCapture(); assert(accepted==0);
    std::cout<<"PASS software trigger and stationary-signal rejection\n";
    reset(); active=true; next=PRE+1; captureDeadline=fakeNow-1; loop();
    assert(!active && !running && accepted==0 && oled.text.find("SAMPLE_LOSS")!=std::string::npos);
    std::cout<<"PASS incomplete capture timeout\n";
    reset(); readyShown=false; rearmAt=fakeNow+500;
    for(size_t i=0;i<PRE;++i) sample(2048);
    sample(3000); assert(!active);
    fakeNow=rearmAt; loop(); assert(readyShown && oled.text.find("Ready")!=std::string::npos);
    std::cout<<"PASS preparing state blocks premature trigger\n";
    reset(); markLoss(); assert(lossHistory==PRE);
    for(size_t i=0;i<PRE;++i) sample(2048);
    assert(lossHistory==0);
    fakeNow=0xFFFFFFF0u; assert(!due(10)); fakeNow=11; assert(due(10));
    std::cout<<"PASS loss-history recovery and millis wraparound\nALL PASSED\n";
}
