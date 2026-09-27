#include <Arduino.h>
#include <math.h>


// 数据协议
constexpr char PROTOCOL_VERSION[] =
    "WOOD_IMPACT_V1";

// 实验信息
constexpr char STICK_ID[] = "WOOD01";

constexpr uint16_t EXPERIMENT_BATCH = 1;
constexpr uint16_t RECLAMP_BATCH = 0;

// 采样参数
constexpr uint32_t SAMPLE_RATE_HZ = 16000;
constexpr size_t SAMPLE_COUNT = 1024;

// ESP32 12位ADC范围
constexpr int ADC_MIN_VALUE = 0;
constexpr int ADC_MAX_VALUE = 4095;
constexpr int ADC_MIDPOINT = 2048;

// 弱信号判断阈值
constexpr int WEAK_SIGNAL_THRESHOLD = 100;

// 数学常数
constexpr float TWO_PI_F =
    6.28318530717958647692f;


// 保存一次敲击的原始信号
uint16_t rawSamples[SAMPLE_COUNT];


struct QualityFlags {
    bool clipped;
    bool sampleLoss;
    bool weakSignal;
};


void generateSimulatedSignal(
    uint16_t impactPointId
) {
    float mainFrequency = 850.0f;
    float amplitude = 900.0f;

    if (impactPointId == 1) {
        mainFrequency = 820.0f;
        amplitude = 820.0f;
    } else if (impactPointId == 2) {
        mainFrequency = 850.0f;
        amplitude = 900.0f;
    } else if (impactPointId == 3) {
        mainFrequency = 880.0f;
        amplitude = 780.0f;
    }

    for (size_t index = 0;
         index < SAMPLE_COUNT;
         ++index) {

        float timeSeconds =
            static_cast<float>(index)
            / static_cast<float>(
                SAMPLE_RATE_HZ
            );

        // 模拟敲击后的衰减
        float envelope =
            expf(-45.0f * timeSeconds);

        float vibration1 =
            amplitude
            * envelope
            * sinf(
                TWO_PI_F
                * mainFrequency
                * timeSeconds
            );

        float vibration2 =
            420.0f
            * envelope
            * sinf(
                TWO_PI_F
                * 1750.0f
                * timeSeconds
            );

        // 模拟传感器噪声
        long noise = random(-12, 13);

        int value = static_cast<int>(
            roundf(
                ADC_MIDPOINT
                + vibration1
                + vibration2
                + static_cast<float>(noise)
            )
        );

        value = constrain(
            value,
            ADC_MIN_VALUE,
            ADC_MAX_VALUE
        );

        rawSamples[index] =
            static_cast<uint16_t>(value);
    }
}


QualityFlags checkSignalQuality(
    size_t actualSampleCount
) {
    QualityFlags flags = {
        false,
        false,
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

        if (value <= ADC_MIN_VALUE
            || value >= ADC_MAX_VALUE) {
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


void printAnomalyFlags(
    const QualityFlags& flags
) {
    Serial.print(
        ",\"anomaly_flags\":["
    );

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
        static_cast<unsigned>(
            EXPERIMENT_BATCH
        ),
        static_cast<unsigned>(
            RECLAMP_BATCH
        ),
        static_cast<unsigned>(
            impactPointId
        ),
        static_cast<unsigned>(
            strikeId
        )
    );

    QualityFlags flags =
        checkSignalQuality(SAMPLE_COUNT);

    Serial.print("{");

    Serial.print(
        "\"protocol_version\":\""
    );
    Serial.print(PROTOCOL_VERSION);
    Serial.print("\"");

    Serial.print(",\"sample_id\":\"");
    Serial.print(sampleId);
    Serial.print("\"");

    Serial.print(",\"stick_id\":\"");
    Serial.print(STICK_ID);
    Serial.print("\"");

    Serial.print(
        ",\"experiment_batch\":"
    );
    Serial.print(EXPERIMENT_BATCH);

    Serial.print(
        ",\"impact_point_id\":"
    );
    Serial.print(impactPointId);

    Serial.print(",\"strike_id\":");
    Serial.print(strikeId);

    Serial.print(
        ",\"reclamp_batch\":"
    );
    Serial.print(RECLAMP_BATCH);

    Serial.print(
        ",\"sample_rate_hz\":"
    );
    Serial.print(SAMPLE_RATE_HZ);

    Serial.print(
        ",\"sample_count\":"
    );
    Serial.print(SAMPLE_COUNT);

    printAnomalyFlags(flags);
    printRawSamples();

    // 一次敲击必须以一行JSON结束
    Serial.println("}");
}


void setup() {
    Serial.begin(115200);

    delay(1000);

    randomSeed(12345);

    // 模拟同一根木头的三个敲击位置
    for (
        uint16_t impactPointId = 1;
        impactPointId <= 3;
        ++impactPointId
    ) {
        generateSimulatedSignal(
            impactPointId
        );

        // 每个位置当前只模拟一次敲击
        outputRecord(
            impactPointId,
            1
        );

        delay(200);
    }
}


void loop() {
    // 当前模拟程序只输出一次
}