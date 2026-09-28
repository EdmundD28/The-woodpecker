import json
import numpy as np
from pathlib import Path

# 读取文件里的第一条敲击记录
data_file = (
    Path(__file__).parent
    / "examples"
    / "WOOD_IMPACT_V1_example.jsonl"
)
with data_file.open(encoding="utf-8") as file:
    record = json.loads(file.readline())

# 取出全部 1024 个原始采样值
raw = record["raw"]

# 求平均值，再让每个采样值减去平均值
mean_value = sum(raw) / len(raw)
centered = [value - mean_value for value in raw]

# 显示处理前后的结果
print("样本编号:", record["sample_id"])
print("原始平均值:", round(mean_value, 4))
print("处理后的平均值:", round(sum(centered) / len(centered), 8))
print("原始前10个值:", raw[:10])
print("去直流后前10个值:", [round(value, 2) for value in centered[:10]])

# 固定缩放：每个值都除以相同的常数
normalized = [value / 2048.0 for value in centered]

print("缩放后前10个值:", [round(value, 4) for value in normalized[:10]])
print("缩放后最小值:", round(min(normalized), 4))
print("缩放后最大值:", round(max(normalized), 4))
print("缩放后数据个数:", len(normalized))

# 转成 float32 数组，排列成“1024 个时间位置 × 1 个通道”
model_input = np.asarray(normalized, dtype=np.float32).reshape(1024, 1)

print("模型输入形状:", model_input.shape)
print("模型输入类型:", model_input.dtype)
print("模型输入占用字节:", model_input.nbytes)
print("所有数值是否有效:", bool(np.isfinite(model_input).all()))