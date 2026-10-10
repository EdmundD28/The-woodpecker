PROTOCOL_VERSION = "WOOD_IMPACT_V1"

# 当前实验固定三个敲击位置
ALLOWED_IMPACT_POINT_IDS = {1, 2, 3}

REQUIRED_FIELDS = {
    "protocol_version",
    "sample_id",
    "stick_id",
    "experiment_batch",
    "impact_point_id",
    "strike_id",
    "reclamp_batch",
    "sample_rate_hz",
    "sample_count",
    "anomaly_flags",
    "raw",
}

ALLOWED_ANOMALY_FLAGS = {
    "CLIPPED",
    "SAMPLE_LOSS",
    "WEAK_SIGNAL",
    "DOUBLE_HIT",
    "TRIGGER_TIMEOUT",
    "MANUAL_INVALID",
}


def validate_integer(
    record: dict,
    field: str,
    minimum: int,
) -> None:
    """检查指定字段是否为满足范围要求的整数。"""

    value = record[field]

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field}必须是整数")

    if value < minimum:
        raise ValueError(
            f"{field}不能小于{minimum}"
        )


def build_sample_id(
    stick_id: str,
    experiment_batch: int,
    reclamp_batch: int,
    impact_point_id: int,
    strike_id: int,
) -> str:
    """根据实验信息生成唯一的样本编号。"""

    return (
        f"{stick_id}"
        f"-E{experiment_batch:02d}"
        f"-R{reclamp_batch:02d}"
        f"-P{impact_point_id:02d}"
        f"-H{strike_id:03d}"
    )


def validate_record(record: dict) -> None:
    """验证一条木材敲击采样记录。"""

    if not isinstance(record, dict):
        raise ValueError(
            "采样记录必须是JSON对象"
        )

    missing_fields = REQUIRED_FIELDS - record.keys()

    if missing_fields:
        raise ValueError(
            f"缺少字段: {sorted(missing_fields)}"
        )

    if record["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(
            "协议版本错误: "
            f"{record['protocol_version']}"
        )

    for field in ("sample_id", "stick_id"):
        if not isinstance(record[field], str):
            raise ValueError(
                f"{field}必须是字符串"
            )

        if not record[field].strip():
            raise ValueError(
                f"{field}不能为空"
            )

    validate_integer(
        record,
        "experiment_batch",
        1,
    )

    validate_integer(
        record,
        "impact_point_id",
        1,
    )

    validate_integer(
        record,
        "strike_id",
        1,
    )

    validate_integer(
        record,
        "reclamp_batch",
        0,
    )

    validate_integer(
        record,
        "sample_rate_hz",
        1,
    )

    validate_integer(
        record,
        "sample_count",
        1,
    )

    impact_point_id = record["impact_point_id"]

    if impact_point_id not in ALLOWED_IMPACT_POINT_IDS:
        raise ValueError(
            "impact_point_id必须是1、2或3"
        )

    expected_sample_id = build_sample_id(
        stick_id=record["stick_id"],
        experiment_batch=record["experiment_batch"],
        reclamp_batch=record["reclamp_batch"],
        impact_point_id=record["impact_point_id"],
        strike_id=record["strike_id"],
    )

    if record["sample_id"] != expected_sample_id:
        raise ValueError(
            "sample_id与实验信息不一致: "
            f"应为{expected_sample_id}"
        )

    raw = record["raw"]

    if not isinstance(raw, list):
        raise ValueError(
            "raw必须是数组"
        )

    if len(raw) != record["sample_count"]:
        raise ValueError(
            "raw长度与sample_count不一致: "
            f"{len(raw)} != "
            f"{record['sample_count']}"
        )

    for index, value in enumerate(raw):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"raw[{index}]必须是整数"
            )

    anomaly_flags = record["anomaly_flags"]

    if not isinstance(anomaly_flags, list):
        raise ValueError(
            "anomaly_flags必须是数组"
        )

    for flag in anomaly_flags:
        if flag not in ALLOWED_ANOMALY_FLAGS:
            raise ValueError(
                f"未知异常标记: {flag}"
            )

    if len(anomaly_flags) != len(set(anomaly_flags)):
        raise ValueError(
            "anomaly_flags中存在重复标记"
        )


if __name__ == "__main__":
    example_record = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_id": "WOOD01-E01-R00-P01-H001",
        "stick_id": "WOOD01",
        "experiment_batch": 1,
        "impact_point_id": 1,
        "strike_id": 1,
        "reclamp_batch": 0,
        "sample_rate_hz": 16000,

        # 这里使用8点只是为了快速测试协议
        "sample_count": 8,

        "anomaly_flags": [],

        "raw": [
            2048,
            2100,
            2180,
            2120,
            2050,
            1990,
            2010,
            2040,
        ],
    }

    validate_record(example_record)

    print("协议验证通过")
    print(
        f"样本编号: "
        f"{example_record['sample_id']}"
    )
    print(
        f"敲击位置: "
        f"P{example_record['impact_point_id']:02d}"
    )
    print(
        f"采样点数: "
        f"{example_record['sample_count']}"
    )