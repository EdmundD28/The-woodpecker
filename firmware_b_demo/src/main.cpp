#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <driver/adc.h>
#include <driver/i2s.h>
#include <math.h>
#include "wood_classifier.h"
#include "touch_permission.h"
#include "oled_anim.h"
#include "signal_quality.h"

static_assert(wood_classifier::scratch_count>=signal_quality::workspace_floats,
              "CNN scratch must hold the single-hit quality FFT");
signal_quality::Policy qualityPolicy=signal_quality::Policy::Venue;
uint8_t shapeRetries=0;

constexpr size_t N=wood_classifier::sample_count, PRE=256, BLOCK=128;
constexpr uint32_t FS=wood_classifier::sample_rate_hz;
constexpr int THRESHOLD=600; // Rehearsal trial: suppress weaker early triggers.
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
// 400 kHz keeps a full frame near 25 ms so the woodpecker animations run smoothly.
constexpr uint32_t OLED_I2C_HZ=400000;
Adafruit_SSD1306 oled(128,64,&Wire,-1,OLED_I2C_HZ,OLED_I2C_HZ);
QueueHandle_t events=nullptr;
uint16_t raw[N], ring[PRE], dma[BLOCK];
size_t head=0, filled=0, next=PRE, discard=256, lossHistory=0;
bool active=false, loss=false, requested=false, running=false;
bool baselineReady=false, finishedBlock=false;
bool readyShown=false;
float baseline=0;
uint32_t rearmAt=0, resumeAt=0, captureDeadline=0;
uint8_t accepted=0;
// Last group's verdict while it waits for a fresh hold: 0 = none or model
// error, otherwise the learned specimen identity (wood 1 or wood 2).
int resultWood=0;
uint8_t resultPhase=0;
uint32_t resultAt=0;
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

// Animations block; once setup is done, keep the GPIO25 liveness output
// ticking while they run (it stays low during boot, as before).
bool setupDone=false;
void animWait(uint32_t ms) {
    while(ms) { uint32_t step=ms>20?20:ms; delay(step); ms-=step; if(setupDone) updateRunStatus(); }
}
void fatal(const char* reason) {
    digitalWrite(RUN_STATUS_PIN,LOW);
    Serial.printf("#FATAL,%s\n",reason); oled_ui::error(oled,reason);
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
    oled_ui::preparing(oled,accepted,"");
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
        oled_ui::cancelled(oled,accepted,"");
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
    if(wanted==GateView::Locked) oled_ui::locked(oled,accepted,"");
    else if(wanted==GateView::Used) oled_ui::releaseFirst(oled,accepted,"");
    else if(wanted==GateView::Arming) oled_ui::arming(oled,accepted,"");
    else oled_ui::ready(oled,accepted,"");
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
    return signal_quality::basic(raw,N,loss);
}
void finishCapture() {
    stopAdc();
    uint32_t processingStart=millis();
    const auto result=signal_quality::evaluate(raw,N,loss,accepted,qualityPolicy,wood_classifier::scratch);
    const char* problem=result.problem;
    Serial.printf("#QUALITY,point=%u,policy=%s,score=%.6f,base=%.6f,limit=%.6f,decision=%s,quality_ms=%lu\n",
                  unsigned(accepted+1),signal_quality::policyName(qualityPolicy),double(result.score),
                  double(result.baseThreshold),double(result.threshold),problem?problem:"PASS",
                  static_cast<unsigned long>(millis()-processingStart));
    if(!problem && !wood_classifier::store_point(raw,accepted)) problem="PREPROCESS_ERROR";
    if(problem) {
        if(!strcmp(problem,"SHAPE_OUTLIER") && shapeRetries<255) ++shapeRetries;
        else if(strcmp(problem,"SHAPE_OUTLIER")) shapeRetries=0;
        Serial.printf("#INVALID,%s,valid=%u/3\n",problem,unsigned(accepted));
        oled_ui::invalid(oled,animWait,accepted,problem,"");
        if(shapeRetries>=3 && !strcmp(problem,"SHAPE_OUTLIER")) {
            oled_ui::setupCheck(oled,accepted);
            Serial.printf("#CHECK_SETUP,point=%u,retries=%u,policy=%s,threshold_unchanged=1\n",
                          unsigned(accepted+1),unsigned(shapeRetries),signal_quality::policyName(qualityPolicy));
        }
        resumeAt=millis()+1200;
    } else {
        shapeRetries=0;
        ++accepted;
        Serial.printf("#TAP,accepted=%u/3,model=%s,preprocess_ms=%lu\n",
                      unsigned(accepted),wood_classifier::mode,
                      static_cast<unsigned long>(millis()-processingStart));
        oled_ui::accepted(oled,animWait,accepted,"");
        if(accepted<3) {
            resumeAt=millis()+200;
        } else {
            oled_ui::thinking(oled,"");
            float scores[2];
            uint32_t inferenceStart=millis();
            if(wood_classifier::classify(scores)) {
                int winner=scores[1]>scores[0]?1:0;
                Serial.printf("#RESULT,wood=%d,score1=%.7f,score2=%.7f,model=%s,version=%s,infer_ms=%lu\n",
                              winner+1,double(scores[0]),double(scores[1]),wood_classifier::mode,
                              wood_classifier::version,static_cast<unsigned long>(millis()-inferenceStart));
                // These weights identify two specimens; they do not diagnose defects.
                resultWood=winner+1;
                oled_ui::result(oled,animWait,resultWood==1,0);
                resultPhase=0; resultAt=millis();
            } else {
                resultWood=0;
                oled_ui::modelError(oled);
                Serial.println("#INVALID,INFERENCE_ERROR,group_discarded=1");
            }
            // Keep the completed group on screen until a fresh one-second hold.
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
        Wire.setClock(OLED_I2C_HZ); Wire.setTimeOut(50);
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
    Serial.println("#BOOT,firmware=B_THREE_POINT_NORMAL_QUALITY_V1");
    traceBoot("OLED_START");
    while(!initialiseDisplay()) {
        Serial.println("#OLED_WAIT,address=0x3C,SDA=21,SCL=22,retrying=1");
        delay(1000);
    }
    traceBoot("OLED_READY");
    oled_ui::boot(oled,animWait,wood_classifier::mode);
    traceBoot("MODEL_ALLOCATE");
    if(!wood_classifier::initialise()) fatal("MODEL_MEMORY_FAILED");
    Serial.printf("#QUALITY_INIT,policy=%s,multiplier=%.2f,reference=%s,workspace_shared_with_cnn=1\n",
                  signal_quality::policyName(qualityPolicy),double(signal_quality::multiplier(qualityPolicy)),
                  quality_reference::sha256);
    traceBoot("MODEL_READY");
    delay(1000);
    oled_ui::preparing(oled,accepted,"");
    if(!initAdc()) fatal("ADC_INIT_FAILED");
    running=true; rearmAt=millis()+500;
    traceBoot("RUNNING"); setupDone=true;
    Serial.printf("#INIT,model=%s,version=%s,FS=%lu,N=%u,PRE=%u,threshold=%d,order=end-middle-end,touch_pin=27,touch_active=%d,hold_ms=1000\n",
                  wood_classifier::mode,wood_classifier::version,static_cast<unsigned long>(FS),
                  unsigned(N),unsigned(PRE),THRESHOLD,TOUCH_ACTIVE_LEVEL);
}
void loop() {
    updateRunStatus();
    if(!pollTouch()) return;
    while(Serial.available()) {
        char c=Serial.read();
        if(c=='?') Serial.printf("#STATUS,stage=%s,running=%d,capturing=%d,touch=%d,accepted=%u,quality=%s,retries=%u\n",
                                bootStage,int(running),int(active),digitalRead(TOUCH_PIN),unsigned(accepted),
                                signal_quality::policyName(qualityPolicy),unsigned(shapeRetries));
        if(c=='V'||c=='v'||c=='S'||c=='s') {
            if(active || accepted==3) Serial.println("#QUALITY_MODE,BUSY,not_changed=1");
            else {
                qualityPolicy=(c=='V'||c=='v')?signal_quality::Policy::Venue:signal_quality::Policy::Strict;
                shapeRetries=0; requested=false;
                if(running) stopAdc();
                oled_ui::qualityMode(oled,accepted,qualityPolicy==signal_quality::Policy::Venue);
                resumeAt=millis()+1200;
                Serial.printf("#QUALITY_MODE,policy=%s,multiplier=%.2f,earlier_points_preserved=%u\n",
                              signal_quality::policyName(qualityPolicy),double(signal_quality::multiplier(qualityPolicy)),unsigned(accepted));
            }
        }
        if((c=='C'||c=='c') && running && !active && readyShown &&
           touchPermission.canStart(millis())) requested=true;
        if((c=='R'||c=='r') && !active && accepted<3) {
            if(running) stopAdc();
            accepted=0; shapeRetries=0; requested=false; resumeAt=millis()+300;
            oled_ui::groupReset(oled,"");
            Serial.println("#RESET,accepted=0/3");
        }
    }
    if(!running) {
        // The final capture consumed its hold, so release/rehold is required.
        if(accepted==3 ? touchPermission.canStart(millis()) : due(resumeAt)) {
            if(accepted==3) {
                // The hold that starts the next group first patches the holes.
                if(resultWood) oled_ui::patch(oled,animWait,resultWood==1);
                accepted=0; shapeRetries=0; resultWood=0;
            }
            restartAdc();
        } else if(accepted==3 && resultWood && due(resultAt)) {
            // Result stays up; its decoration takes turns with a "hold" hint.
            static const uint16_t idleMs[4]={500,500,700,700};
            oled_ui::resultIdle(oled,resultWood==1,resultPhase);
            resultAt=millis()+idleMs[resultPhase]; resultPhase=(resultPhase+1)%4;
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
            stopAdc(); active=false;
            Serial.println("#INVALID,SAMPLE_LOSS");
            oled_ui::invalid(oled,animWait,accepted,"SAMPLE_LOSS","");
            resumeAt=millis()+1200;
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

