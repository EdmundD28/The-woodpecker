import json
from pathlib import Path

import numpy as np
from protocol import validate_record

data_file = (
    Path(__file__).parent
    / "examples"
    / "WOOD_IMPACT_V1_example.jsonl"
)

passed = 0
rejected = 0
seen_ids = set()
accepted_inputs = []
accepted_ids = []

with data_file.open(encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)

            # 检查记录是否符合队友提供的数据协议
            validate_record(record)

            # 检查是否满足我们约定的模型输入要求
            if record["sample_rate_hz"] != 16000:
                raise ValueError("采样率必须为16000 Hz")

            if record["sample_count"] != 1024:
                raise ValueError("采样点数必须为1024")

            if record["anomaly_flags"]:
                raise ValueError(
                    f"记录带有异常标记：{record['anomaly_flags']}"
                )

            sample_id = record["sample_id"]
            if sample_id in seen_ids:
                raise ValueError("样本编号重复")
            seen_ids.add(sample_id)

            # 与预览程序相同：去直流 → 固定缩放
            raw = record["raw"]
            mean_value = sum(raw) / len(raw)
            normalized = [
                (value - mean_value) / 2048.0
                for value in raw
            ]

            # 整理成模型输入
            model_input = np.asarray(
                normalized, dtype=np.float32
            ).reshape(1024, 1)

            if not np.isfinite(model_input).all():
                raise ValueError("处理结果存在无效数值")
            
            accepted_inputs.append(model_input)
            accepted_ids.append(sample_id)

            passed += 1
            print(
                f"通过：{sample_id}，"
                f"形状={model_input.shape}，"
                f"类型={model_input.dtype}"
            )

        except (ValueError, TypeError) as error:
            rejected += 1
            print(f"拒绝：第{line_number}行，原因：{error}")

print()
print("处理完成")
print("通过记录数:", passed)
print("拒绝记录数:", rejected)

# 将所有通过检查的输入合并并保存
if accepted_inputs:
    batch = np.stack(accepted_inputs, axis=0)

    output_folder = Path(__file__).parent / "collected_data"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "model_inputs.npz"

    np.savez(
        output_path,
        inputs=batch,
        sample_ids=np.asarray(accepted_ids),
        preprocessing_version=np.asarray("TIME_V0_1"),
    )

    print("保存后的整体形状:", batch.shape)
    print("保存位置:", output_path)
else:
    print("没有可保存的模型输入")