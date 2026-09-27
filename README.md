# The-woodpecker

木材敲击振动采集与分类项目。项目通过敲击木材触发振动，由 ESP32 采集原始信号；后续对信号进行预处理、FFT 和一维小型卷积神经网络推理，实现四种木材的质量分类。

## 2026-09-27 阶段目标

本阶段不依赖真实开发板和传感器，目标是先确定软件结构和数据接口：

- 建立固件 A、固件 B 和电脑采集工具三个相互独立、可继续开发的工程。
- 冻结包含原始信号、样本编号、敲击编号、重新装夹批次、采样率、采样点数和异常标记的数据协议。
- 使用模拟数据跑通“ESP32 输出格式 → 电脑读取与校验 → JSONL 文件保存”。

以上流程已使用模拟数据验证通过。

## 工程结构

```text
The-woodpecker/
├─ firmware_a_sample/                 # 固件 A：实验采样版
│  ├─ platformio.ini
│  └─ src/main.cpp
├─ firmware_b_demo/                   # 固件 B：现场展示版
│  ├─ platformio.ini
│  └─ src/main.cpp
├─ pc_collector/                      # 电脑采集工具
│  ├─ collector.py                    # 读取、验证并保存 JSONL
│  ├─ device_simulator.py             # 模拟 ESP32 输出
│  ├─ protocol.py                     # 协议构造与验证
│  └─ examples/
│     ├─ protocol.md                  # 完整协议说明
│     └─ WOOD_IMPACT_V1_example.jsonl # 冻结的示例文件
└─ wood-impact-system.code-workspace
```

### 固件 A：实验采样版

最终用于触发采样、输出原始振动信号、检测异常并将数据发送给电脑。当前版本使用模拟振动数据，拿到真实传感器后应主要替换采样函数，尽量保持已经冻结的输出协议不变。

### 固件 B：现场展示版

最终用于采样、固定预处理、分类器推理、连续三次敲击多数表决和 OLED 显示。当前只建立了可编译的独立工程，尚未加入 FFT、神经网络和 OLED 功能。

### 电脑采集工具

接收一行一条的 JSON 数据，使用 `protocol.py` 校验字段、编号和原始信号长度，随后保存为 JSONL 文件。当前可由 `device_simulator.py` 提供模拟输入；后续再增加真实串口输入。

## 冻结的数据协议

- 协议版本：`WOOD_IMPACT_V1`
- 状态：`FROZEN`
- 编码：UTF-8
- 格式：JSON Lines；一次敲击对应一行 JSON

每条记录包含以下字段：

| 字段 | 含义 |
|---|---|
| `protocol_version` | 协议版本，固定为 `WOOD_IMPACT_V1` |
| `sample_id` | 唯一样本编号 |
| `stick_id` | 木头编号 |
| `experiment_batch` | 实验批次 |
| `impact_point_id` | 敲击位置编号，目前为 1（头部）、2（中部）、3（尾部） |
| `strike_id` | 同一位置的第几次敲击 |
| `reclamp_batch` | 重新装夹批次；首次装夹为 0 |
| `sample_rate_hz` | 采样率，当前为 16000 Hz |
| `sample_count` | 每次敲击的采样点数，当前为 1024 |
| `anomaly_flags` | 异常标记数组，无异常时为空数组 |
| `raw` | 原始时域采样值数组，长度必须等于 `sample_count` |

样本编号示例：

```text
WOOD01-E01-R00-P01-H001
```

含义为：第 1 根木头、第 1 次实验、未重新装夹、第 1 个敲击位置、该位置第 1 次敲击。

允许的异常标记及其对应情况如下：

| 异常标记 | 对应情况 | 处理建议 |
|---|---|---|
| `CLIPPED`（削顶） | 一个或多个采样值达到 ADC 下限或上限，说明信号可能超出量程并发生削顶失真 | 检查传感器增益、偏置和输入保护；该条数据通常不用于正常训练 |
| `SAMPLE_LOSS`（丢样） | 实际获得的采样点数不等于 `sample_count`，或者采样过程中出现丢点、数据中断 | 检查采样定时和缓冲区；建议重新采样 |
| `WEAK_SIGNAL`（弱信号） | 信号峰峰值低于弱信号阈值，可能是敲击过轻、传感器接触不良或安装位置不合适 | 检查敲击力度和传感器安装；建议重新敲击 |
| `DOUBLE_HIT`（重复敲击） | 一次采样窗口中检测到两个明显的敲击事件，例如敲击工具发生二次接触 | 本次记录不能代表一次独立敲击，建议重新采样 |
| `TRIGGER_TIMEOUT`（触发超时） | 在规定等待时间内没有检测到有效敲击触发，因而没有获得有效采样 | 作为采样流程状态记录，不放入正常训练集；检查触发阈值后重新敲击 |
| `MANUAL_INVALID`（人工判定无效） | 实验人员确认该次操作无效，例如敲错位置、木头移动、触碰传感器或受到明显外界干扰 | 由电脑端或实验记录人员添加，ESP32 不会自动产生；该条数据不用于正常训练 |

没有检测到异常时，`anomaly_flags` 必须为空数组 `[]`。同一条记录可以同时包含多个异常标记，例如：

```json
"anomaly_flags":["CLIPPED","DOUBLE_HIT"]
```

当前固件 A 已包含 `CLIPPED`、`SAMPLE_LOSS` 和 `WEAK_SIGNAL` 的基础检查逻辑；`DOUBLE_HIT` 将在接入真实传感器后通过算法检测；`TRIGGER_TIMEOUT` 属于采样流程状态；`MANUAL_INVALID` 仅由电脑端或实验记录人员在人工复核后添加。当前模拟器生成的正常示例使用空数组 `[]`。

协议冻结后不得直接改变字段名称、含义或数据类型。如需进行不兼容修改，应升级为 `WOOD_IMPACT_V2`。完整说明见 `pc_collector/examples/protocol.md`。

## 快速验证

### 1. 编译固件 A

```powershell
cd .\firmware_a_sample
platformio run
```

### 2. 编译固件 B

```powershell
cd .\firmware_b_demo
platformio run
```

### 3. 运行模拟采集流程

需要 Python 3，无需安装第三方 Python 库。

```powershell
cd .\pc_collector
py .\device_simulator.py --repeats 1 --interval 0.2 | py .\collector.py
```

正确结果是收到三条记录，`impact_point_id` 分别为 1、2、3，并在以下目录生成新的 JSONL 文件：

```text
pc_collector/collected_data/session_*.jsonl
```

`collected_data` 是本地实验输出目录，不提交到 Git。需要提交的固定示例位于 `pc_collector/examples/WOOD_IMPACT_V1_example.jsonl`。

## 当前采样与接口假设

- 开发板：ESP32-WROOM-32D 开发板。
- 开发框架：Arduino Framework，使用 VS Code + PlatformIO。
- 串口速率：115200 baud。
- 采样率：16000 Hz。
- 每次敲击：1024 点，对应 64 ms 信号。
- 当前模拟值按照 ESP32 12 位 ADC 范围生成：0～4095，中点约为 2048。
- 真实传感器型号、输出类型、ADC/I2S 接口、GPIO 和模拟前端电路尚未冻结，需要电气负责人确认。
- 如果使用压电或其他模拟传感器，必须确认输入电压范围、偏置、限幅保护、放大和抗混叠要求，不能将可能为负或过压的信号直接接入 ESP32。

## 交付对象

| 接收人 | 交付内容 | 用途 |
|---|---|---|
| 罗 | `protocol.md`、协议示例 JSONL、`protocol.py` 和 `collector.py` | 开发训练数据读取流水线 |
| 李 | 样本编号、实验批次、敲击位置、重新装夹和异常标记规则 | 制定实验规程 |
| 戴 | 采样率、采样点数、ADC 与串口假设，以及尚待确认的传感器接口问题 | 核对传感器和电气接口 |

## 后续开发

1. 确定传感器型号、电气接口和安装方式。
2. 将固件 A 的模拟信号生成函数替换为真实采样，同时保持 `WOOD_IMPACT_V1` 输出格式。
3. 按实验规程采集四种木材的数据并训练分类模型。
4. 固化与训练端完全一致的预处理和 FFT 参数。
5. 将模型部署到固件 B，加入三次多数表决、低置信度处理。
