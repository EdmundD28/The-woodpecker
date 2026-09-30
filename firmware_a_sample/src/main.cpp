#include <Arduino.h>
#include <esp_timer.h>
#include <driver/adc.h>
#include <driver/i2s.h>

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

// 硬件连续采样；主循环按块读取，不再依赖analogRead的软件耗时。
constexpr i2s_port_t ADC_PORT = I2S_NUM_0;
constexpr adc1_channel_t ADC_CHANNEL = ADC1_CHANNEL_6;
constexpr size_t DMA_BLOCK_SAMPLES = 128;
uint16_t dmaBlock[DMA_BLOCK_SAMPLES];
QueueHandle_t adcEventQueue = nullptr;
bool captureFinishedInBlock = false;
bool timingDiagnostics = false;
uint32_t dmaErrorCount = 0;
size_t lossHistoryRemaining = 0;
size_t startupDiscardRemaining = 256;
uint64_t dmaRateStartUs = 0;
uint32_t dmaRateSamples = 0;
uint32_t measuredRateHz = 0;

// 实时预览每64点发送其中最大偏差点：16000 / 64 = 250 Hz
constexpr uint16_t PREVIEW_DECIMATION = 64;

// ESP32 12位ADC及真机调试参数
constexpr int ADC_MIN_VALUE = 0;
constexpr int ADC_MAX_VALUE = 4095;
constexpr int CLIP_LOW = 10;
constexpr int CLIP_HIGH = 4085;
constexpr int TRIGGER_THRESHOLD = 300;
constexpr int WEAK_SIGNAL_THRESHOLD = 100;

// 避免一次敲击的余振被识别成多次敲击
constexpr uint32_t REARM_DELAY_MS = 500;

// SAMPLE_LOSS现在依据DMA溢出、读取失败和通道错误，保留真实异常判定。

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
bool monitorEnabled = false;
uint16_t previewSampleCounter = 0;
uint16_t previewPeakSample = 0;
int previewPeakDifference = 0;

uint32_t rearmUntilMs = 0;


struct QualityFlags {
    bool clipped;
    bool sampleLoss;
    bool weakSignal;
};


void markDmaLoss() {
    ++dmaErrorCount;
    if (captureActive) sampleLossDetected = true;
    lossHistoryRemaining = PRE_TRIGGER_COUNT;
}

void checkDmaEvents() {
    i2s_event_t event;
    while (xQueueReceive(adcEventQueue, &event, 0) == pdTRUE) {
        if (event.type == I2S_EVENT_RX_Q_OVF || event.type == I2S_EVENT_DMA_ERROR) {
            markDmaLoss();
        }
    }
}

bool initialiseContinuousAdc() {
    if (adc1_config_width(ADC_WIDTH_BIT_12) != ESP_OK ||
        adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_12) != ESP_OK) return false;
    i2s_config_t config{};
    config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN);
    config.sample_rate = SAMPLE_RATE_HZ;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
    config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_MSB;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = 8;
    config.dma_buf_len = DMA_BLOCK_SAMPLES;
    config.use_apll = true;
    if (i2s_driver_install(ADC_PORT, &config, 32, &adcEventQueue) != ESP_OK) return false;
    return i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL) == ESP_OK &&
        i2s_adc_enable(ADC_PORT) == ESP_OK;
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
    sampleLossDetected = lossHistoryRemaining != 0;
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
    // 先停止采样再传输；约半秒的JSON发送不再造成缓冲溢出。
    i2s_adc_disable(ADC_PORT);
    i2s_stop(ADC_PORT);
    checkDmaEvents();
    uint16_t strikeId = nextStrikeId[currentImpactPointId - 1]++;

    outputRecord(currentImpactPointId, strikeId);
    if (timingDiagnostics) {
        Serial.printf("#DMA,rate_hz=%u,errors=%u\n", unsigned(measuredRateHz), unsigned(dmaErrorCount));
    }
    Serial.flush();

    captureActive = false;
    sampleLossDetected = false;
    preTriggerWriteIndex = 0;
    preTriggerValidCount = 0;
    baselineInitialized = false;
    rearmUntilMs = millis() + REARM_DELAY_MS;
    previewSampleCounter = 0;
    previewPeakDifference = 0;
    lossHistoryRemaining = 0;
    // 清除停止前已排队的数据，下一条记录只使用重新启动后的样本。
    size_t staleBytes = 0;
    do {
        i2s_read(ADC_PORT, dmaBlock, sizeof(dmaBlock), &staleBytes, 0);
    } while (staleBytes != 0);
    xQueueReset(adcEventQueue);
    startupDiscardRemaining = 256;
    dmaRateStartUs = 0;
    dmaRateSamples = 0;
    if (i2s_start(ADC_PORT) != ESP_OK || i2s_adc_enable(ADC_PORT) != ESP_OK) {
        Serial.println("#FATAL,ADC_RESTART_FAILED");
        while (true) delay(1000);
    }
    captureFinishedInBlock = true;
}


void outputPreviewSample(
    uint16_t sample,
    int differenceFromBaseline
) {
    if (!monitorEnabled || captureActive) {
        return;
    }

    if (
        previewSampleCounter == 0
        || differenceFromBaseline > previewPeakDifference
    ) {
        previewPeakSample = sample;
        previewPeakDifference = differenceFromBaseline;
    }

    ++previewSampleCounter;

    if (previewSampleCounter < PREVIEW_DECIMATION) {
        return;
    }

    previewSampleCounter = 0;

    // 独立预览格式，不属于WOOD_IMPACT_V1训练记录：
    // P,timestamp_us,sample,baseline,difference
    char line[80];
    int length = snprintf(line, sizeof(line), "P,%lu,%u,%.1f,%d\n",
        static_cast<unsigned long>(micros()), unsigned(previewPeakSample),
        baseline, previewPeakDifference);
    // 预览可跳帧，但绝不能阻塞真实采样数据的读取。
    if (length > 0 && length < int(sizeof(line)) && Serial.availableForWrite() >= length) {
        Serial.write(reinterpret_cast<const uint8_t*>(line), length);
    }

    previewPeakDifference = 0;
}


void readSerialCommands() {
    while (Serial.available() > 0) {
        char command = static_cast<char>(Serial.read());
        if (command == 'D') timingDiagnostics = true;
        if (command == 'd') timingDiagnostics = false;

        // 发送1、2、3选择当前敲击位置
        if (command >= '1' && command <= '3') {
            currentImpactPointId =
                static_cast<uint16_t>(command - '0');
        }

        // 发送C执行一次软件触发
        if (command == 'C' || command == 'c') {
            softwareTriggerRequested = true;
        }

        // M开启实时预览，m关闭实时预览
        if (command == 'M') {
            monitorEnabled = true;
            previewSampleCounter = 0;
            previewPeakDifference = 0;
        } else if (command == 'm') {
            monitorEnabled = false;
        }
    }
}


void takeOneSample(uint16_t sample) {

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
    if (lossHistoryRemaining > 0) --lossHistoryRemaining;
    outputPreviewSample(sample, distanceFromBaseline);
}


void setup() {
    Serial.begin(115200);

    if (!initialiseContinuousAdc()) {
        Serial.println("#FATAL,ADC_INITIALISATION_FAILED");
        while (true) delay(1000);
    }
}


void loop() {
    readSerialCommands();

    size_t bytesRead = 0;
    esp_err_t result = i2s_read(ADC_PORT, dmaBlock, sizeof(dmaBlock), &bytesRead, pdMS_TO_TICKS(100));
    checkDmaEvents();
    if (result != ESP_OK || bytesRead == 0 || bytesRead % sizeof(uint16_t) != 0) {
        markDmaLoss();
        return;
    }
    size_t count = bytesRead / sizeof(uint16_t);
    uint64_t nowUs = esp_timer_get_time();
    if (dmaRateStartUs == 0) {
        dmaRateStartUs = nowUs;
        dmaRateSamples = 0;
    } else {
        dmaRateSamples += count;
        if (nowUs - dmaRateStartUs >= 1000000) {
            measuredRateHz = uint64_t(dmaRateSamples) * 1000000 / (nowUs - dmaRateStartUs);
            dmaRateStartUs = nowUs;
            dmaRateSamples = 0;
        }
    }
    captureFinishedInBlock = false;
    for (size_t index = 0; index < count; ++index) {
        if (startupDiscardRemaining > 0) {
            --startupDiscardRemaining;
            continue;
        }
        uint16_t word = dmaBlock[index];
        if ((word >> 12) != unsigned(ADC_CHANNEL)) {
            markDmaLoss();
            continue;
        }
        takeOneSample(word & 0x0FFF);
        if (captureFinishedInBlock) break;
    }
}
