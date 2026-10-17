#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <driver/adc.h>
#include <driver/i2s.h>
#include <math.h>
#include "wood_classifier.h"
#include "touch_permission.h"

constexpr size_t N=wood_classifier::sample_count, PRE=256, BLOCK=128;
constexpr uint32_t FS=wood_classifier::sample_rate_hz;
constexpr int THRESHOLD=500; // Match the firmware A used for this training run.
constexpr uint8_t OLED_SDA=21, OLED_SCL=22, OLED_ADDRESS=0x3C;
constexpr uint8_t TOUCH_PIN=27;
constexpr int TOUCH_ACTIVE_LEVEL=HIGH; // Change to LOW for an active-low module.
constexpr uint8_t RUN_STATUS_PIN=25; // Temporary diagnostic output; leave unloaded.
bool runStatusHigh=false;
uint32_t runStatusAt=0;
void updateRunStatus() {
    uint32_t now=millis();
    if(uint32_t(now-runStatusAt)>=1000) {
        runStatusAt=now; runStatusHigh=!runStatusHigh;
        digitalWrite(RUN_STATUS_PIN,runStatusHigh?HIGH:LOW);
    }
}
constexpr adc1_channel_t CHANNEL=ADC1_CHANNEL_6; // Protected piezo -> GPIO34
constexpr i2s_port_t PORT=I2S_NUM_0;
// Use 100 kHz for both commands and frame transfers, including cold startup.
Adafruit_SSD1306 oled(128,64,&Wire,-1,100000,100000);
QueueHandle_t events=nullptr;
uint16_t raw[N], ring[PRE], dma[BLOCK];
size_t head=0, filled=0, next=PRE, discard=256, lossHistory=0;
bool active=false, loss=false, requested=false, running=false;
bool baselineReady=false, finishedBlock=false;
bool readyShown=false;
float baseline=0;
uint32_t rearmAt=0, resumeAt=0, captureDeadline=0;
uint8_t accepted=0;
TouchPermission touchPermission;
enum class GateView { Unknown, Locked, Arming, Ready, Used };
GateView gateView=GateView::Unknown;
const char* bootStage="START";
void traceBoot(const char* stage) {
    bootStage=stage;
    Serial.printf("#BOOT_STAGE,%s,ms=%lu\n",stage,static_cast<unsigned long>(millis()));
#if defined(ARDUINO_ARCH_ESP32)
    Serial.printf("#HEAP,free=%u,largest=%u\n",unsigned(ESP.getFreeHeap()),unsigned(ESP.getMaxAllocHeap()));
#endif
}
bool due(uint32_t deadline) { return int32_t(millis()-deadline)>=0; }
const char* nextPoint() {
    return accepted==0?"Tap first endpoint":accepted==1?"Tap middle point":"Tap other endpoint";
}

void screen(const char* state,const char* detail="",int cls=0) {
    oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(0,0); oled.printf("WOOD %s CNN\n",wood_classifier::mode);
    oled.println(state); oled.printf("Valid taps: %u/3\n",unsigned(accepted));
    oled.println(detail);
    if(cls) { oled.setTextSize(2); oled.printf("WOOD: %d",cls); }
    oled.display();
}
void fatal(const char* reason) {
    digitalWrite(RUN_STATUS_PIN,LOW);
    Serial.printf("#FATAL,%s\n",reason); screen("ERROR",reason);
    while(true) delay(1000);
}
void markLoss() { if(active) loss=true; lossHistory=PRE; }
void checkEvents() {
    i2s_event_t event;
    while(xQueueReceive(events,&event,0)==pdTRUE)
        if(event.type==I2S_EVENT_RX_Q_OVF || event.type==I2S_EVENT_DMA_ERROR) markLoss();
}
bool initAdc() {
    traceBoot("ADC_CONFIG");
    if(adc1_config_width(ADC_WIDTH_BIT_12)!=ESP_OK ||
       adc1_config_channel_atten(CHANNEL,ADC_ATTEN_DB_12)!=ESP_OK) return false;
    i2s_config_t cfg{};
    cfg.mode=static_cast<i2s_mode_t>(I2S_MODE_MASTER|I2S_MODE_RX|I2S_MODE_ADC_BUILT_IN);
    cfg.sample_rate=FS; cfg.bits_per_sample=I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format=I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format=I2S_COMM_FORMAT_STAND_MSB;
    cfg.intr_alloc_flags=ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count=8; cfg.dma_buf_len=BLOCK; cfg.use_apll=true;
    traceBoot("ADC_DRIVER_INSTALL");
    if(i2s_driver_install(PORT,&cfg,32,&events)!=ESP_OK) return false;
    traceBoot("ADC_SET_MODE");
    if(i2s_set_adc_mode(ADC_UNIT_1,CHANNEL)!=ESP_OK) return false;
    traceBoot("ADC_ENABLE");
    if(i2s_adc_enable(PORT)!=ESP_OK) return false;
    traceBoot("ADC_READY");
    return true;
}
void stopAdc() {
    if(i2s_adc_disable(PORT)!=ESP_OK || i2s_stop(PORT)!=ESP_OK) fatal("ADC_STOP_FAILED");
    running=false; checkEvents();
}
void restartAdc() {
    // No OLED transfers during ADC acquisition; drop data queued before pause.
    size_t stale=0;
    // At most the configured eight DMA buffers; never hang while draining.
    for(unsigned i=0;i<8;++i) {
        if(i2s_read(PORT,dma,sizeof(dma),&stale,0)!=ESP_OK || !stale) break;
    }
    xQueueReset(events); head=filled=0; discard=PRE; lossHistory=0;
    active=loss=baselineReady=requested=false;
    readyShown=false;
    gateView=GateView::Unknown;
    screen("Preparing","Please wait...");
    if(i2s_start(PORT)!=ESP_OK || i2s_adc_enable(PORT)!=ESP_OK) fatal("ADC_RESTART_FAILED");
    running=true; rearmAt=millis()+500;
}
bool pollTouch() {
    bool wasHeld=touchPermission.held();
    touchPermission.update(digitalRead(TOUCH_PIN)==TOUCH_ACTIVE_LEVEL,millis());
    if(wasHeld!=touchPermission.held()) {
        // Never let a new hold inherit pretrigger samples or a queued C command.
        head=filled=0; baselineReady=false; requested=false; readyShown=false;
        discard=PRE;
    }
    if(active && !touchPermission.held()) {
        stopAdc(); active=requested=false; readyShown=false;
        screen("Tap cancelled","Hold touch to retry");
        Serial.printf("#CANCELLED,TOUCH_RELEASED,valid=%u/3\n",unsigned(accepted));
        resumeAt=millis()+700; finishedBlock=true;
        return false;
    }
    return true;
}
void updateGateScreen() {
    if(active || !running) return;
    GateView wanted=GateView::Locked;
    if(!touchPermission.held() && digitalRead(TOUCH_PIN)==TOUCH_ACTIVE_LEVEL)
        wanted=GateView::Used; // Power-on while touched requires a release first.
    if(touchPermission.held()) {
        if(touchPermission.consumed()) wanted=GateView::Used;
        else if(touchPermission.canStart(millis()) && due(rearmAt) &&
                (readyShown || (baselineReady && filled==PRE && !discard))) wanted=GateView::Ready;
        else wanted=GateView::Arming;
    }
    if(wanted==gateView) return;
    stopAdc();
    const char* state=wanted==GateView::Locked?"Hold touch to arm":
                      wanted==GateView::Used?"Release, then hold":
                      wanted==GateView::Arming?"Hold for 1 second":"Ready - keep holding";
    screen(state,nextPoint());
    // Discard everything queued during display I/O, including before this hold.
    size_t stale=0;
    for(unsigned i=0;i<8;++i) {
        if(i2s_read(PORT,dma,sizeof(dma),&stale,0)!=ESP_OK || !stale) break;
    }
    xQueueReset(events); head=filled=0; lossHistory=0; baselineReady=false;
    // Ready keeps its status while these 256 fresh pretrigger points refill.
    discard=0; readyShown=wanted==GateView::Ready; gateView=wanted;
    if(i2s_start(PORT)!=ESP_OK || i2s_adc_enable(PORT)!=ESP_OK) fatal("ADC_RESTART_FAILED");
    running=true;
    Serial.printf("#TOUCH,state=%s,next=%u/3,pin=%u,level=%d\n",
                  wanted==GateView::Locked?"LOCKED":wanted==GateView::Used?"RELEASE_REQUIRED":
                  wanted==GateView::Arming?"ARMING":"READY",unsigned(accepted+1),
                  unsigned(TOUCH_PIN),digitalRead(TOUCH_PIN));
}
const char* quality() {
    uint16_t low=4095,high=0; bool clipped=false;
    for(auto value:raw) {
        if(value<=10 || value>=4085) clipped=true;
        if(value<low) low=value;
        if(value>high) high=value;
    }
    if(loss) return "SAMPLE_LOSS";
    if(clipped) return "CLIPPED";
    if(high-low<100) return "WEAK_SIGNAL";
    return nullptr;
}
void finishCapture() {
    stopAdc();
    const char* problem=quality();
    uint32_t processingStart=millis();
    if(!problem && !wood_classifier::store_point(raw,accepted)) problem="PREPROCESS_ERROR";
    if(problem) {
        Serial.printf("#INVALID,%s,valid=%u/3\n",problem,unsigned(accepted));
        screen("Invalid - retry",problem); resumeAt=millis()+1500;
    } else {
        ++accepted;
        Serial.printf("#TAP,accepted=%u/3,model=%s,preprocess_ms=%lu\n",
                      unsigned(accepted),wood_classifier::mode,
                      static_cast<unsigned long>(millis()-processingStart));
        if(accepted<3) {
            screen("Tap accepted",nextPoint()); resumeAt=millis()+700;
        } else {
            screen("Classifying...","Three points ready");
            float scores[2];
            uint32_t inferenceStart=millis();
            if(wood_classifier::classify(scores)) {
                int winner=scores[1]>scores[0]?1:0;
                screen("Three-point result","Known woods 1 / 2",winner+1);
                Serial.printf("#RESULT,wood=%d,score1=%.7f,score2=%.7f,model=%s,version=%s,infer_ms=%lu\n",
                              winner+1,double(scores[0]),double(scores[1]),wood_classifier::mode,
                              wood_classifier::version,static_cast<unsigned long>(millis()-inferenceStart));
            } else {
                screen("Model error","Restart group (R)");
                Serial.println("#INVALID,INFERENCE_ERROR,group_discarded=1");
            }
            resumeAt=millis()+5000;
        }
    }
    active=requested=false; finishedBlock=true;
}
void sample(uint16_t value) {
    if(!pollTouch()) return;
    if(active) { raw[next++]=value; if(next==N) finishCapture(); return; }
    int delta=abs(int(value)-int(baseline));
    if(readyShown && baselineReady && filled==PRE && due(rearmAt) &&
       touchPermission.canStart(millis()) && (requested || delta>=THRESHOLD)) {
        for(size_t i=0;i<PRE;++i) raw[i]=ring[(head+i)%PRE];
        raw[PRE]=value; next=PRE+1; loss=lossHistory!=0;
        touchPermission.consume();
        active=true; requested=false; captureDeadline=millis()+600; return;
    }
    if(!baselineReady) { baseline=value; baselineReady=true; }
    else baseline+=(float(value)-baseline)/64.0f;
    ring[head]=value; head=(head+1)%PRE;
    if(filled<PRE) ++filled;
    if(lossHistory) --lossHistory;
}
bool initialiseDisplay() {
    // Four-pin modules have no MCU-controlled reset. Their power-on reset can
    // finish after the ESP32 starts; an early NACK must not permanently halt us.
    delay(500);
    for(unsigned attempt=1;attempt<=6;++attempt) {
        Wire.end();
        Wire.begin(OLED_SDA,OLED_SCL);
        Wire.setClock(100000); Wire.setTimeOut(50);
        Wire.beginTransmission(OLED_ADDRESS);
        int status=Wire.endTransmission();
        bool initialised=false;
        if(status==0 && oled.begin(SSD1306_SWITCHCAPVCC,OLED_ADDRESS,false,false)) {
            // ACK only confirms a responding bus device. Reapply the complete
            // init after settling, then check the bus again (not pixel proof).
            delay(150);
            initialised=oled.begin(SSD1306_SWITCHCAPVCC,OLED_ADDRESS,false,false);
            Wire.beginTransmission(OLED_ADDRESS);
            status=Wire.endTransmission();
        }
        Serial.printf("#OLED_BOOT,attempt=%u,i2c_status=%d,init=%d\n",
                      attempt,status,int(initialised));
        if(initialised && status==0) return true;
        delay(500);
    }
    return false;
}
void setup() {
    pinMode(RUN_STATUS_PIN,OUTPUT);
    digitalWrite(RUN_STATUS_PIN,LOW); runStatusHigh=false; runStatusAt=millis();
    readyShown=false;
    gateView=GateView::Unknown;
    pinMode(TOUCH_PIN,TOUCH_ACTIVE_LEVEL==HIGH?INPUT_PULLDOWN:INPUT_PULLUP);
    touchPermission=TouchPermission{};
    pollTouch();
    Serial.begin(115200);
    Serial.println("#BOOT,firmware=B_TOUCH_OLED_RETRY_V3_RUN_PIN25");
    traceBoot("OLED_START");
    while(!initialiseDisplay()) {
        Serial.println("#OLED_WAIT,address=0x3C,SDA=21,SCL=22,retrying=1");
        delay(1000);
    }
    traceBoot("OLED_READY");
    screen("Initialising",wood_classifier::mode);
    traceBoot("MODEL_ALLOCATE");
    if(!wood_classifier::initialise()) fatal("MODEL_MEMORY_FAILED");
    traceBoot("MODEL_READY");
    delay(1000);
    screen("Preparing","Please wait...");
    if(!initAdc()) fatal("ADC_INIT_FAILED");
    running=true; rearmAt=millis()+500;
    traceBoot("RUNNING");
    Serial.printf("#INIT,model=%s,version=%s,FS=%lu,N=%u,PRE=%u,threshold=%d,order=end-middle-end,touch_pin=27,touch_active=%d,hold_ms=1000\n",
                  wood_classifier::mode,wood_classifier::version,static_cast<unsigned long>(FS),
                  unsigned(N),unsigned(PRE),THRESHOLD,TOUCH_ACTIVE_LEVEL);
}
void loop() {
    updateRunStatus();
    if(!pollTouch()) return;
    while(Serial.available()) {
        char c=Serial.read();
        if(c=='?') Serial.printf("#STATUS,stage=%s,running=%d,capturing=%d,touch=%d,accepted=%u\n",
                                bootStage,int(running),int(active),digitalRead(TOUCH_PIN),unsigned(accepted));
        if((c=='C'||c=='c') && running && !active && readyShown &&
           touchPermission.canStart(millis())) requested=true;
        if((c=='R'||c=='r') && !active) {
            if(running) stopAdc();
            accepted=0; requested=false; resumeAt=millis()+300;
            screen("Group reset","Start at an endpoint");
            Serial.println("#RESET,accepted=0/3");
        }
    }
    if(!running) {
        if(due(resumeAt)) {
            if(accepted==3) accepted=0;
            restartAdc();
        }
        delay(1); return;
    }
    updateGateScreen();
    size_t bytes=0;
    esp_err_t result=i2s_read(PORT,dma,sizeof(dma),&bytes,pdMS_TO_TICKS(100));
    checkEvents();
    if(result!=ESP_OK || bytes==0 || bytes%sizeof(uint16_t) || (active && due(captureDeadline))) {
        markLoss();
        if(active) {
            stopAdc(); active=false; screen("Invalid - retry","SAMPLE_LOSS");
            Serial.println("#INVALID,SAMPLE_LOSS"); resumeAt=millis()+1500;
        }
        return;
    }
    finishedBlock=false;
    for(size_t i=0;i<bytes/sizeof(uint16_t);++i) {
        if(discard) { --discard; continue; }
        if((dma[i]>>12)!=unsigned(CHANNEL)) { markLoss(); continue; }
        sample(dma[i]&0xFFF); if(finishedBlock) break;
    }
}

