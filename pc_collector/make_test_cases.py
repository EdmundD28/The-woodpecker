import json
from copy import deepcopy
from pathlib import Path

base_folder = Path(__file__).parent

folder = base_folder / "collected_data"
folder.mkdir(parents=True, exist_ok=True)

source = (
    base_folder
    / "examples"
    / "WOOD_IMPACT_V1_example.jsonl"
)

with source.open(encoding="utf-8") as file:
    normal = json.loads(file.readline())

# 测试1：保留一条正常记录
cases = [normal]

# 测试2：采样率错误
wrong_rate = deepcopy(normal)
wrong_rate["sample_rate_hz"] = 8000
cases.append(wrong_rate)

# 测试3：少了一个采样点
missing_point = deepcopy(normal)
missing_point["raw"].pop()
cases.append(missing_point)

# 测试4：记录被标记为弱信号
flagged_signal = deepcopy(normal)
flagged_signal["anomaly_flags"] = ["WEAK_SIGNAL"]
cases.append(flagged_signal)

# 保存到单独的测试文件
output = folder / "interface_test_cases.jsonl"

with output.open("w", encoding="utf-8") as file:
    for record in cases:
        file.write(json.dumps(record) + "\n")

print("测试文件已生成:", output)
print("包含1条正常记录和3条错误记录")