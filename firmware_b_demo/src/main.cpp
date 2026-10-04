#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <driver/adc.h>
#include <driver/i2s.h>
#include <math.h>

constexpr size_t N=1024, PRE=256, BLOCK=128;
constexpr uint32_t FS=16000;
constexpr int THRESHOLD=300;
constexpr uint8_t OLED_SDA=21, OLED_SCL=22, OLED_ADDRESS=0x3C;
constexpr adc1_channel_t CHANNEL=ADC1_CHANNEL_6; // Protected piezo -> GPIO34
constexpr i2s_port_t PORT=I2S_NUM_0;
Adafruit_SSD1306 oled(128,64,&Wire,-1);
QueueHandle_t events=nullptr;
uint16_t raw[N], ring[PRE], dma[BLOCK];
float modelInput[N]; // TIME_V0_1 float32 [1024,1]
size_t head=0, filled=0, next=PRE, discard=256, lossHistory=0;
bool active=false, loss=false, requested=false, running=false;
bool baselineReady=false, finishedBlock=false;
bool readyShown=false;
float baseline=0;
uint32_t rearmAt=0, resumeAt=0, captureDeadline=0;
uint8_t votes[4]={}, accepted=0;
bool due(uint32_t deadline) { return int32_t(millis()-deadline)>=0; }

void screen(const char* state,const char* detail="",int cls=0) {
    oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(0,0); oled.println("WOOD DEMO");
    oled.println(state); oled.printf("Valid taps: %u/3\n",unsigned(accepted));
    oled.println(detail);
    if(cls) { oled.setTextSize(2); oled.printf("DEMO: %d",cls); }
    oled.display();
}
void fatal(const char* reason) {
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
    if(adc1_config_width(ADC_WIDTH_BIT_12)!=ESP_OK ||
       adc1_config_channel_atten(CHANNEL,ADC_ATTEN_DB_12)!=ESP_OK) return false;
    i2s_config_t cfg{};
    cfg.mode=static_cast<i2s_mode_t>(I2S_MODE_MASTER|I2S_MODE_RX|I2S_MODE_ADC_BUILT_IN);
    cfg.sample_rate=FS; cfg.bits_per_sample=I2S_BITS_PER_SAMPLE_16BIT;
    cfg.channel_format=I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format=I2S_COMM_FORMAT_STAND_MSB;
    cfg.intr_alloc_flags=ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count=8; cfg.dma_buf_len=BLOCK; cfg.use_apll=true;
    return i2s_driver_install(PORT,&cfg,32,&events)==ESP_OK &&
        i2s_set_adc_mode(ADC_UNIT_1,CHANNEL)==ESP_OK && i2s_adc_enable(PORT)==ESP_OK;
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
    screen("Preparing","Please wait...");
    if(i2s_start(PORT)!=ESP_OK || i2s_adc_enable(PORT)!=ESP_OK) fatal("ADC_RESTART_FAILED");
    running=true; rearmAt=millis()+500;
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
bool preprocess() {
    // Exact lhf preprocess_batch.py operation: mean removal then fixed /2048.
    uint32_t sum=0; for(auto value:raw) sum+=value;
    float mean=float(sum)/N;
    for(size_t i=0;i<N;++i) {
        modelInput[i]=(float(raw[i])-mean)/2048.0f;
        if(!isfinite(modelInput[i])) return false;
    }
    return true;
}
int classifyPlaceholder(const float* input) {
    // TEST_ONLY arbitrary amplitude bins, not a trained wood classifier.
    float peak=0;
    for(size_t i=0;i<N;++i) peak=fmaxf(peak,fabsf(input[i]));
    return peak<0.20f?0:peak<0.40f?1:peak<0.60f?2:3;
}
void finishCapture() {
    stopAdc();
    const char* problem=quality();
    if(!problem && !preprocess()) problem="PREPROCESS_ERROR";
    if(problem) {
        Serial.printf("#INVALID,%s,valid=%u/3\n",problem,unsigned(accepted));
        screen("Invalid - retry",problem); resumeAt=millis()+1500;
    } else {
        int cls=classifyPlaceholder(modelInput); ++votes[cls]; ++accepted;
        Serial.printf("#DEMO,tap=%u/3,class_id=%d,display_class=%d,model=TEST_ONLY\n",
                      unsigned(accepted),cls,cls+1);
        if(accepted<3) {
            screen("Tap accepted","Next tap after Ready",cls+1); resumeAt=millis()+700;
        } else {
            int winner=-1; for(int i=0;i<4;++i) if(votes[i]>=2) winner=i;
            if(winner<0) screen("No majority","Uncertain - retry");
            else screen("Final DEMO result","NOT real prediction",winner+1);
            Serial.printf("#DEMO_FINAL,display_class=%d,status=%s\n",winner+1,
                          winner<0?"NO_MAJORITY":"DEMO_ONLY");
            resumeAt=millis()+5000;
        }
    }
    active=requested=false; finishedBlock=true;
}
void sample(uint16_t value) {
    if(active) { raw[next++]=value; if(next==N) finishCapture(); return; }
    int delta=abs(int(value)-int(baseline));
    if(readyShown && baselineReady && filled==PRE && due(rearmAt) && (requested || delta>=THRESHOLD)) {
        for(size_t i=0;i<PRE;++i) raw[i]=ring[(head+i)%PRE];
        raw[PRE]=value; next=PRE+1; loss=lossHistory!=0;
        active=true; requested=false; captureDeadline=millis()+250; return;
    }
    if(!baselineReady) { baseline=value; baselineReady=true; }
    else baseline+=(float(value)-baseline)/64.0f;
    ring[head]=value; head=(head+1)%PRE;
    if(filled<PRE) ++filled;
    if(lossHistory) --lossHistory;
}
void setup() {
    readyShown=false;
    Serial.begin(115200); Wire.begin(OLED_SDA,OLED_SCL); Wire.setTimeOut(50);
    Wire.beginTransmission(OLED_ADDRESS);
    if(Wire.endTransmission()!=0 || !oled.begin(SSD1306_SWITCHCAPVCC,OLED_ADDRESS,false,false)) {
        Serial.println("#FATAL,OLED_NOT_FOUND,address=0x3C,SDA=21,SCL=22");
        while(true) delay(1000);
    }
    screen("Initialising","Classifier: TEST_ONLY"); delay(1000);
    screen("Preparing","Please wait...");
    if(!initAdc()) fatal("ADC_INIT_FAILED");
    running=true; rearmAt=millis()+500;
    Serial.println("#INIT,DEMO_ONLY,FS=16000,N=1024,PRE=256,threshold=300");
}
void loop() {
    while(Serial.available()) {
        char c=Serial.read();
        if((c=='C'||c=='c') && running && !active) requested=true;
    }
    if(!running) {
        if(due(resumeAt)) {
            if(accepted==3) { accepted=0; for(auto& vote:votes) vote=0; }
            restartAdc();
        }
        delay(1); return;
    }
    if(!readyShown && baselineReady && filled==PRE && due(rearmAt)) {
        // Refresh only outside a capture. Discard frames acquired during the
        // display transfer, then refill the prebuffer before allowing a trigger.
        stopAdc();
        screen("Ready","Tap wood; C=soft tap");
        size_t stale=0;
        for(unsigned i=0;i<8;++i) {
            if(i2s_read(PORT,dma,sizeof(dma),&stale,0)!=ESP_OK || !stale) break;
        }
        xQueueReset(events); filled=head=0; lossHistory=0;
        if(i2s_start(PORT)!=ESP_OK || i2s_adc_enable(PORT)!=ESP_OK) fatal("ADC_RESTART_FAILED");
        running=true; readyShown=true;
        Serial.println("#READY,DEMO_ONLY");
    }
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

