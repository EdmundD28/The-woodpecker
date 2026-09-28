import math


def build_output(
    sample_id,
    scores=None,
    model_version=None,
    labels_ready=False,
    anomaly_flags=None,
    confidence_threshold=None,
):
    """把模型分数整理为统一输出。"""
    flags = list(anomaly_flags or [])

    result = {
        "interface_version": "WOOD_MODEL_OUTPUT_DRAFT_V0_1",
        "sample_id": sample_id,
        "model_version": model_version,
        "status": "NOT_READY",
        "scores": None,
        "class_id": None,
        "confidence": None,
        "anomaly_flags": flags,
        "reason": "MODEL_OR_LABEL_MAPPING_NOT_READY",
    }

    # 有异常标记的信号不进行正常分类
    if flags:
        result["status"] = "INVALID_SIGNAL"
        result["reason"] = "INPUT_HAS_ANOMALY_FLAGS"
        return result

    # 模型或类别映射尚未就绪
    if model_version is None or not labels_ready:
        return result

    # 检查四类分数：必须有限、在0～1之间、总和接近1
    try:
        values = [float(value) for value in scores]
        valid = (
            len(values) == 4
            and all(
                math.isfinite(value) and 0 <= value <= 1
                for value in values
            )
            and abs(sum(values) - 1.0) <= 0.00001
        )
    except (TypeError, ValueError, OverflowError):
        valid = False

    if not valid:
        result["status"] = "INFERENCE_ERROR"
        result["reason"] = "INVALID_CLASS_SCORES"
        return result

    if confidence_threshold is not None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("置信度阈值必须在0～1之间")

    confidence = max(values)
    result["scores"] = values
    result["confidence"] = confidence

    # 最高分并列时，不强行选择类别
    if values.count(confidence) > 1:
        result["status"] = "LOW_CONFIDENCE"
        result["reason"] = "TIED_TOP_SCORES"
        return result

    # 仅在配置阈值后启用低置信度拒绝
    if (
        confidence_threshold is not None
        and confidence < confidence_threshold
    ):
        result["status"] = "LOW_CONFIDENCE"
        result["reason"] = "BELOW_CONFIDENCE_THRESHOLD"
        return result

    result["status"] = "OK"
    result["class_id"] = values.index(confidence)
    result["reason"] = None
    return result


if __name__ == "__main__":
    sample_id = "WOOD01-E01-R00-P01-H001"

    # 以下全部使用人工测试数据，不是真实模型预测
    def test_result(scores, **kwargs):
        return build_output(
            sample_id,
            scores=scores,
            model_version="TEST_ONLY",
            labels_ready=True,
            **kwargs,
        )

    normal = test_result([0.08, 0.72, 0.15, 0.05])
    assert normal["status"] == "OK"
    assert normal["class_id"] == 1
    assert normal["confidence"] == 0.72
    print("通过：正常分数能够得到类别编号和最高分")

    not_ready = build_output(sample_id)
    assert not_ready["status"] == "NOT_READY"
    assert not_ready["class_id"] is None
    print("通过：模型未就绪时不输出分类")

    invalid = test_result(
        [0.08, 0.72, 0.15, 0.05],
        anomaly_flags=["WEAK_SIGNAL"],
    )
    assert invalid["status"] == "INVALID_SIGNAL"
    assert invalid["scores"] is None
    print("通过：异常信号不输出分类")

    # 0.8只是测试阈值，不是实际项目确定的阈值
    low = test_result(
        [0.08, 0.72, 0.15, 0.05],
        confidence_threshold=0.8,
    )
    assert low["status"] == "LOW_CONFIDENCE"
    assert low["class_id"] is None
    print("通过：低于测试阈值时不接受分类")

    tied = test_result([0.4, 0.4, 0.1, 0.1])
    assert tied["status"] == "LOW_CONFIDENCE"
    assert tied["class_id"] is None
    print("通过：最高分并列时不强行选择类别")

    bad = test_result([0.8, 0.8, 0.1, 0.1])
    assert bad["status"] == "INFERENCE_ERROR"
    assert bad["class_id"] is None
    print("通过：分数总和错误时报告推理错误")

    print("输出接口6项测试全部通过")