# 木材敲击振动数据协议

## 协议状态

- 协议名称：WOOD IMPACT
- 协议版本：WOOD_IMPACT_V1
- 当前状态：FROZEN
- 数据编码：UTF-8
- 数据格式：JSON Lines
- 一次敲击对应一行JSON
- 每行以换行符结束

协议冻结后，不允许直接修改字段名称、字段含义或数据类型。

如果以后必须进行不兼容修改，应升级为：

WOOD_IMPACT_V2

## 数据流向

数据生产端：

- 固件A实验采样版
- device_simulator.py模拟器

数据接收端：

- collector.py电脑采集工具
- 后续模型训练流水线

数据流：

固件A或模拟器
→ 输出JSON行
→ collector.py读取
→ protocol.py验证
→ 保存为JSONL文件

## 完整记录格式

```json
{
  "protocol_version": "WOOD_IMPACT_V1",
  "sample_id": "WOOD01-E01-R00-P01-H001",
  "stick_id": "WOOD01",
  "experiment_batch": 1,
  "impact_point_id": 1,
  "strike_id": 1,
  "reclamp_batch": 0,
  "sample_rate_hz": 16000,
  "sample_count": 4,
  "anomaly_flags": [],
  "raw": [2048, 2100, 2180, 2120]
}
```

上面的短示例使用4个采样点，便于阅读。当前实际模拟采样参数为16000 Hz、每次1024点；正式示例见 `WOOD_IMPACT_V1_example.jsonl`。
