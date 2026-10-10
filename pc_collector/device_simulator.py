import argparse
import json
import math
import random
import time

from protocol import PROTOCOL_VERSION
from protocol import build_sample_id
from protocol import validate_record


SAMPLE_RATE_HZ = 16000
SAMPLE_COUNT = 1024
PRE_TRIGGER_COUNT = 256
PREVIEW_RATE_HZ = 250
PREVIEW_DURATION_SECONDS = 2.0
PREVIEW_IMPACT_TIME_SECONDS = 1.0

ADC_MIDPOINT = 2048
ADC_MIN = 0
ADC_MAX = 4095

STICK_ID = "WOOD01"
EXPERIMENT_BATCH = 1

# 目前不进行重新装夹
RECLAMP_BATCH = 0

# 每根木头固定三个敲击位置
IMPACT_POINT_IDS = (1, 2, 3)


def generate_raw_signal(
    impact_point_id: int,
    strike_id: int,
) -> list[int]:
    """生成一个指定位置的模拟敲击振动信号。"""

    random.seed(
        1000
        + impact_point_id * 100
        + strike_id
    )

    # 不同敲击位置产生少量频率和幅值差异
    main_frequency = {
        1: 820.0,
        2: 850.0,
        3: 880.0,
    }[impact_point_id]

    amplitude = {
        1: 820.0,
        2: 900.0,
        3: 780.0,
    }[impact_point_id]

    raw = []

    for index in range(SAMPLE_COUNT):
        time_after_trigger = (
            index - PRE_TRIGGER_COUNT
        ) / SAMPLE_RATE_HZ

        vibration_1 = 0.0
        vibration_2 = 0.0

        if time_after_trigger >= 0.0:
            # 模拟触发后的衰减振动；前256点只包含静止噪声
            envelope = math.exp(-45.0 * time_after_trigger)

            vibration_1 = (
                amplitude
                * envelope
                * math.sin(
                    2.0
                    * math.pi
                    * main_frequency
                    * time_after_trigger
                )
            )

            vibration_2 = (
                420.0
                * envelope
                * math.sin(
                    2.0
                    * math.pi
                    * 1750.0
                    * time_after_trigger
                )
            )

        noise = random.gauss(0.0, 12.0)

        value = round(
            ADC_MIDPOINT
            + vibration_1
            + vibration_2
            + noise
        )

        # 限制在ESP32 12位ADC范围内
        value = max(
            ADC_MIN,
            min(ADC_MAX, value),
        )

        raw.append(value)

    return raw


def output_preview_sequence(
    impact_point_id: int,
    strike_id: int,
    start_timestamp_us: int,
    realtime: bool,
) -> int:
    """输出两秒预览，在中间模拟一次明显敲击。"""

    random.seed(
        9000
        + impact_point_id * 100
        + strike_id
    )

    point_count = round(
        PREVIEW_RATE_HZ * PREVIEW_DURATION_SECONDS
    )
    period_us = round(1_000_000 / PREVIEW_RATE_HZ)
    amplitude = {
        1: 700.0,
        2: 850.0,
        3: 620.0,
    }[impact_point_id]

    timestamp_us = start_timestamp_us

    for index in range(point_count):
        time_s = index / PREVIEW_RATE_HZ
        noise = random.gauss(0.0, 10.0)
        excursion = 0.0

        if time_s >= PREVIEW_IMPACT_TIME_SECONDS:
            time_after_impact = (
                time_s - PREVIEW_IMPACT_TIME_SECONDS
            )
            envelope = math.exp(-12.0 * time_after_impact)
            direction = 1.0 if index % 2 == 0 else -1.0
            excursion = direction * amplitude * envelope

        sample = round(ADC_MIDPOINT + noise + excursion)
        sample = max(ADC_MIN, min(ADC_MAX, sample))
        difference = abs(sample - ADC_MIDPOINT)

        print(
            f"P,{timestamp_us},{sample},"
            f"{ADC_MIDPOINT:.1f},{difference}",
            flush=True,
        )

        timestamp_us += period_us

        if realtime:
            time.sleep(1.0 / PREVIEW_RATE_HZ)

    return timestamp_us


def build_record(
    impact_point_id: int,
    strike_id: int,
) -> dict:
    """建立一条符合协议的模拟采样记录。"""

    raw = generate_raw_signal(
        impact_point_id=impact_point_id,
        strike_id=strike_id,
    )

    sample_id = build_sample_id(
        stick_id=STICK_ID,
        experiment_batch=EXPERIMENT_BATCH,
        reclamp_batch=RECLAMP_BATCH,
        impact_point_id=impact_point_id,
        strike_id=strike_id,
    )

    record = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_id": sample_id,
        "stick_id": STICK_ID,
        "experiment_batch": EXPERIMENT_BATCH,
        "impact_point_id": impact_point_id,
        "strike_id": strike_id,
        "reclamp_batch": RECLAMP_BATCH,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_count": SAMPLE_COUNT,
        "anomaly_flags": [],
        "raw": raw,
    }

    validate_record(record)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="模拟三个位置的木材敲击数据"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每个位置重复敲击多少次",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="每条记录之间等待多少秒",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="在每条完整记录前输出两秒250 Hz实时预览",
    )
    parser.add_argument(
        "--preview-fast",
        action="store_true",
        help="生成预览但不按真实时间等待，用于自动测试",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError(
            "repeats必须大于等于1"
        )

    if args.interval < 0:
        raise ValueError(
            "interval不能小于0"
        )

    first_record = True
    preview_timestamp_us = 0

    for impact_point_id in IMPACT_POINT_IDS:
        for strike_id in range(
            1,
            args.repeats + 1,
        ):
            if not first_record:
                time.sleep(args.interval)

            record = build_record(
                impact_point_id=impact_point_id,
                strike_id=strike_id,
            )

            if args.preview:
                preview_timestamp_us = output_preview_sequence(
                    impact_point_id=impact_point_id,
                    strike_id=strike_id,
                    start_timestamp_us=preview_timestamp_us,
                    realtime=not args.preview_fast,
                )

            output_line = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            # 标准输出只能放JSON，方便collector读取
            print(output_line, flush=True)

            first_record = False


if __name__ == "__main__":
    main()
