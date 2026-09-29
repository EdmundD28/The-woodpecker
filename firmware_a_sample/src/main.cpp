#include <Arduino.h>
#include <esp_timer.h>

// 数据协议（保持WOOD_IMPACT_V1字段不变）
constexpr char PROTOCOL_VERSION[] = "WOOD_IMPACT_V1";

// 实验信息
constexpr char STICK_ID[] = "WOOD01";
constexpr uint16_t EXPERIMENT_BATCH = 1;
constexpr uint16_t RECLAMP_BATCH = 0;

// DFR0052 Analog经过保护电路后接到ADC1引脚GPIO34
constexpr uint8_t PIEZO_PIN = 34;

// 固定采样参数
constexpr uint32_t SAMPLE_RATE_HZ = 16000;
constexpr size_t SAMPLE_COUNT = 1024;
constexpr size_t PRE_TRIGGER_COUNT = 256;
constexpr size_t POST_TRIGGER_COUNT = SAMPLE_COUNT - PRE_TRIGGER_COUNT;

// 16kHz的周期是62.5us，交替等待62us和63us以保持平均16kHz
constexpr uint32_t SAMPLE_PERIOD_SHORT_US = 62;
constexpr uint32_t SAMPLE_PERIOD_LONG_US = 63;

// ESP32 12位ADC及真机调试参数
constexpr int ADC_MIN_VALUE = 0;
constexpr int ADC_MAX_VALUE = 4095;
constexpr int CLIP_LOW = 10;
constexpr int CLIP_HIGH = 4085;
constexpr int TRIGGER_THRESHOLD = 200;
constexpr int WEAK_SIGNAL_THRESHOLD = 100;

// 避免一次敲击的余振被识别成多次敲击
constexpr uint32_t REARM_DELAY_MS = 500;

// 如果采样时刻落后计划超过两个采样周期，标记丢样
constexpr uint32_t SAMPLE_LOSS_LIMIT_US = 125;

uint16_t rawSamples[SAMPLE_COUNT];
uint16_t preTriggerBuffer[PRE_TRIGGER_COUNT];

size_t preTriggerWriteIndex = 0;
size_t preTriggerValidCount = 0;
size_t postTriggerWriteIndex = PRE_TRIGGER_COUNT;

uint16_t currentImpactPointId = 1;
uint16_t nextStrikeId[3] = {1, 1, 1};

float baseline = 0.0f;
bool baselineInitialized = false;
bool captureActive = false;
bool softwareTriggerRequested = false;
bool sampleLossDetected = false;

uint64_t nextSampleTimeUs = 0;
bool useLongSamplePeriod = false;
uint32_t rearmUntilMs = 0;


struct QualityFlags {
    bool clipped;
    bool sampleLoss;
    bool weakSignal;
};


uint32_t nextSamplePeriodUs() {
    useLongSamplePeriod = !useLongSamplePeriod;

    return useLongSamplePeriod
        ? SAMPLE_PERIOD_LONG_US
        : SAMPLE_PERIOD_SHORT_US;
}


void scheduleNextSample() {
    nextSampleTimeUs += nextSamplePeriodUs();
}


void resetSamplingSchedule() {
    useLongSamplePeriod = false;
    nextSampleTimeUs =
        static_cast<uint64_t>(esp_timer_get_time())
        + nextSamplePeriodUs();
}


void updateBaseline(uint16_t sample) {
    if (!baselineInitialized) {
        baseline = static_cast<float>(sample);
        baselineInitialized = true;
        return;
    }

    // 慢速跟踪静态偏置，不追随瞬间敲击
    baseline +=
        (static_cast<float>(sample) - baseline)
        / 64.0f;
}


void storePreTriggerSample(uint16_t sample) {
    preTriggerBuffer[preTriggerWriteIndex] = sample;
    preTriggerWriteIndex =
        (preTriggerWriteIndex + 1)
        % PRE_TRIGGER_COUNT;

    if (preTriggerValidCount < PRE_TRIGGER_COUNT) {
        ++preTriggerValidCount;
    }
}


void copyPreTriggerSamples() {
    // writeIndex指向环形缓冲区中最旧的数据
    for (size_t index = 0;
         index < PRE_TRIGGER_COUNT;
         ++index) {
        size_t sourceIndex =
            (preTriggerWriteIndex + index)
            % PRE_TRIGGER_COUNT;

        rawSamples[index] = preTriggerBuffer[sourceIndex];
    }
}


void beginCapture(uint16_t triggerSample) {
    copyPreTriggerSamples();

    // 第256号元素保存触发当下的采样值；之后还会采767点
    rawSamples[PRE_TRIGGER_COUNT] = triggerSample;
    postTriggerWriteIndex = PRE_TRIGGER_COUNT + 1;
    captureActive = true;
    softwareTriggerRequested = false;
}


QualityFlags checkSignalQuality(size_t actualSampleCount) {
    QualityFlags flags = {
        false,
        sampleLossDetected,
        false,
    };

    if (actualSampleCount != SAMPLE_COUNT) {
        flags.sampleLoss = true;
    }

    uint16_t minimumValue = ADC_MAX_VALUE;
    uint16_t maximumValue = ADC_MIN_VALUE;

    for (size_t index = 0;
         index < actualSampleCount;
         ++index) {
        uint16_t value = rawSamples[index];

        if (value <= CLIP_LOW || value >= CLIP_HIGH) {
            flags.clipped = true;
        }

        if (value < minimumValue) {
            minimumValue = value;
        }

        if (value > maximumValue) {
            maximumValue = value;
        }
    }

    int signalRange =
        static_cast<int>(maximumValue)
        - static_cast<int>(minimumValue);

    if (signalRange < WEAK_SIGNAL_THRESHOLD) {
        flags.weakSignal = true;
    }

    return flags;
}


void printAnomalyFlags(const QualityFlags& flags) {
    Serial.print(",\"anomaly_flags\":[");

    bool firstFlag = true;

    if (flags.clipped) {
        Serial.print("\"CLIPPED\"");
        firstFlag = false;
    }

    if (flags.sampleLoss) {
        if (!firstFlag) {
            Serial.print(",");
        }
        Serial.print("\"SAMPLE_LOSS\"");
        firstFlag = false;
    }

    if (flags.weakSignal) {
        if (!firstFlag) {
            Serial.print(",");
        }
        Serial.print("\"WEAK_SIGNAL\"");
    }

    Serial.print("]");
}


void printRawSamples() {
    Serial.print(",\"raw\":[");

    for (size_t index = 0;
         index < SAMPLE_COUNT;
         ++index) {
        Serial.print(rawSamples[index]);

        if (index + 1 < SAMPLE_COUNT) {
            Serial.print(",");
        }
    }

    Serial.print("]");
}


void outputRecord(
    uint16_t impactPointId,
    uint16_t strikeId
) {
    char sampleId[48];

    snprintf(
        sampleId,
        sizeof(sampleId),
        "%s-E%02u-R%02u-P%02u-H%03u",
        STICK_ID,
        static_cast<unsigned>(EXPERIMENT_BATCH),
        static_cast<unsigned>(RECLAMP_BATCH),
        static_cast<unsigned>(impactPointId),
        static_cast<unsigned>(strikeId)
    );

    QualityFlags flags = checkSignalQuality(SAMPLE_COUNT);

    Serial.print("{");
    Serial.print("\"protocol_version\":\"");
    Serial.print(PROTOCOL_VERSION);
    Serial.print("\"");
    Serial.print(",\"sample_id\":\"");
    Serial.print(sampleId);
    Serial.print("\"");
    Serial.print(",\"stick_id\":\"");
    Serial.print(STICK_ID);
    Serial.print("\"");
    Serial.print(",\"experiment_batch\":");
    Serial.print(EXPERIMENT_BATCH);
    Serial.print(",\"impact_point_id\":");
    Serial.print(impactPointId);
    Serial.print(",\"strike_id\":");
    Serial.print(strikeId);
    Serial.print(",\"reclamp_batch\":");
    Serial.print(RECLAMP_BATCH);
    Serial.print(",\"sample_rate_hz\":");
    Serial.print(SAMPLE_RATE_HZ);
    Serial.print(",\"sample_count\":");
    Serial.print(SAMPLE_COUNT);

    printAnomalyFlags(flags);
    printRawSamples();

    // 一次敲击必须以一行JSON结束
    Serial.println("}");
}


void finishCapture() {
    uint16_t strikeId = nextStrikeId[currentImpactPointId - 1]++;

    outputRecord(currentImpactPointId, strikeId);

    captureActive = false;
    sampleLossDetected = false;
    preTriggerWriteIndex = 0;
    preTriggerValidCount = 0;
    baselineInitialized = false;
    rearmUntilMs = millis() + REARM_DELAY_MS;
    resetSamplingSchedule();
}


void readSerialCommands() {
    while (Serial.available() > 0) {
        char command = static_cast<char>(Serial.read());

        // 发送1、2、3选择当前敲击位置
        if (command >= '1' && command <= '3') {
            currentImpactPointId =
                static_cast<uint16_t>(command - '0');
        }

        // 发送C执行一次软件触发
        if (command == 'C' || command == 'c') {
            softwareTriggerRequested = true;
        }
    }
}


void takeOneSample() {
    uint16_t sample = static_cast<uint16_t>(
        analogRead(PIEZO_PIN)
    );

    if (captureActive) {
        rawSamples[postTriggerWriteIndex++] = sample;

        if (postTriggerWriteIndex >= SAMPLE_COUNT) {
            finishCapture();
        }
        return;
    }

    bool bufferReady =
        preTriggerValidCount == PRE_TRIGGER_COUNT;

    int distanceFromBaseline = abs(
        static_cast<int>(sample)
        - static_cast<int>(baseline)
    );

    bool automaticTrigger =
        bufferReady
        && millis() >= rearmUntilMs
        && distanceFromBaseline >= TRIGGER_THRESHOLD;

    bool softwareTrigger =
        bufferReady
        && softwareTriggerRequested;

    if (automaticTrigger || softwareTrigger) {
        beginCapture(sample);
        return;
    }

    updateBaseline(sample);
    storePreTriggerSample(sample);
}


void setup() {
    Serial.begin(115200);

    analogReadResolution(12);
    analogSetPinAttenuation(PIEZO_PIN, ADC_11db);

    resetSamplingSchedule();
}


void loop() {
    readSerialCommands();

    uint64_t nowUs =
        static_cast<uint64_t>(esp_timer_get_time());

    if (nowUs < nextSampleTimeUs) {
        return;
    }

    uint64_t latenessUs = nowUs - nextSampleTimeUs;

    if (captureActive
        && latenessUs > SAMPLE_LOSS_LIMIT_US) {
        sampleLossDetected = true;
    }

    scheduleNextSample();
    takeOneSample();
}
