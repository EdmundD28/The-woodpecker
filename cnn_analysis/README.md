# 信号分析、神经网络训练与质量检查

当前远端模型交付见[质量优化版与优化频域权重](releases/README.md)，默认清单为`current_release.json`，预测使用`python -m cnn_analysis.release_predict --records 新三点记录.jsonl`。采用质量优化版与轻微增强后的频域权重；90%保底、95%目标针对质量合格数据，异常混入仅作压力对照；频带融合不采用。本轮更新电脑端模型交付，固件B与开发板未更新。

从仓库根目录运行命令。无需连接ESP32，采集工作仍由王的pc_collector负责。

## 安装与第一次运行

当前默认输入为 **8000Hz、2400点、300毫秒、单通道**，触发点索引256，即32毫秒预触发。
保留完整窗口，不补点、不截断、不重采样；减整窗均值，再固定除2048。
训练适配版本升级为TIME_V0_2 / SPECTRUM_V0_2，原始记录仍使用WOOD_IMPACT_V1。
旧TIME_V0_1模型及配置仍可明确选择16kHz、1024点；新旧记录和权重不会静默混用。

| 模型 | 输入 | 参数量 | 浮点权重字节 | 推理中间缓冲字节 | 卷积和全连接乘加次数 |
|---|---|---:|---:|---:|---:|
| 时域 | 2400×1 | 2370 | 9480 | 76800 | 931264 |
| 频谱 | 1201×1 | 11970 | 47880 | 38656 | 477512 |

频谱频点间隔为8000/2400≈3.333Hz，最高频率4000Hz；频谱参考预处理用直接DFT，额外存储19200字节Hann窗，未计入上表运算数。
ESP32的总内存和时延仍需实测；采集固件和展示固件没有在本次适配中更新。
默认学习率从0.003调整为0.0003：扩展到300毫秒后，频谱幅值较小，旧步长在固定模拟回归中使网络停在两类各猜一半；较小步长恢复了可学习性。真实数据仍需用验证集检查，模拟结果不代表实测准确率。

从本克隆目录读取现有采集数据示例：

```powershell
py -m cnn_analysis prepare --data ../The-woodpecker/pc_collector/collected_data/20261017_1737_WOOD01_P01_N020.jsonl --out cnn_analysis/runs/current_capture_01
```

读取不需要猜测类别。正式训练仍需确认两类标签和真实来源；只有一类时不生成两分类模型。

需要Python 3.10+；Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r cnn_analysis/requirements.txt
.\.venv\Scripts\python.exe -m cnn_analysis demo --out cnn_analysis/runs/demo_01
```

程序生成288次模拟敲击、12个独立装夹组、两类、三个位置；两套CNN分别跑通整个流程。
频率差异人为设置，用于验证程序，不代表实际木棍差异。所有模拟模型与图表明确标DEMO_ONLY。
输出目录必须是新目录，重复运行换名称，避免覆盖旧结果。

## 两套CNN备选（均保留）

| 备选 | 输入 | 导出目录 |
|---|---|---|
| time | 完整2400点TIME_V0_2时域信号，float32 `[2400,1]` | 输出目录/time |
| spectrum | 同一完整信号减均值除2048后，Hann窗、RFFT幅值除2400，float32 `[1201,1]` | 输出目录/spectrum |

时域版保留全部触发前与触发后采样，不另截窗；频谱版包含DC到Nyquist全部1201频点。
频谱不是波形PNG，也不是125Hz预览；没有保留相位、不做log或逐样本峰值归一化，不额外将单边幅值乘2。
两套模型均用三层小型1D CNN，分别训练权重。时域版用全局平均池化；频谱版Flatten保留频率位置，再接两类输出。
频谱峰的位置可能是类别依据，因此避免用全局平均池化抹掉它。两套不能混用输入或模型文件。
默认train/demo同时训练两套；--input-mode time或--input-mode spectrum可单独跑。
两套使用同一数据、标签、划分策略与随机种子；candidate_comparison.json并排显示结果，不自动淘汰任何备选。
最终选择依据验证集、稳定性和部署代价；不要根据测试集挑版本后仍把该测试集称为未参与选择的独立验收。

## 已经有王的文件：先确认能读取

JSONL是一行一次敲击；JSON是单次敲击。两种均可读，同时提供时相同sample_id且内容相同只算一次。
同编号内容不同会停止整次操作，需核对实验编号；不会静默选择一个。

```powershell
.\.venv\Scripts\python.exe -m cnn_analysis prepare --data cnn_analysis/runs/demo_01/demo_data/demo.jsonl --out cnn_analysis/runs/interface_01
```

上述模拟数据仅用于检查读取。pc_collector/examples里的旧16kHz、1024点示例与当前默认配置不匹配，会被拒收；不能作为当前实测数据。
实际文件直接传给--data，或放到cnn_analysis/data/real目录。
多个文件用空格分隔；带空格的路径用引号。

## 李交真实数据后：填写类别和来源

复制config.json为config_real.json，在stick_labels中填写经实际确认的对应关系。
例如 `{"WOOD01":"solid","WOOD02":"hollow"}` **只展示格式，不说明这两根的实际类别**。
若不同木棍也是同一类别可分别映射；不要在不同实验里把同一stick_id用于不同木棍。

复制data/manifest_template.csv为manifest_real.csv，每条sample_id填一行：

| 字段 | 填什么 |
|---|---|
| sample_id | 采集文件内的完整编号 |
| label | solid或hollow；若config已配置对应木棍，可留空 |
| source | real或synthetic，按真实来源填写 |
| release_method | manual或mechanical，按实际释放方式填写 |
| electrical_verified | 戴确认该数据电气质量后填true，否则false |

manifest是旁表，不改变王冻结的采样协议。未填写来源视为unknown；未确认类别不猜测。
不要把模拟器数据登记成real。真实和模拟数据不能混在正式训练中。

## 信号分析（10月5日）

```powershell
.\.venv\Scripts\python.exe -m cnn_analysis analyze --config cnn_analysis/config_real.json --data pc_collector/collected_data --manifest cnn_analysis/data/manifest_real.csv --out cnn_analysis/runs/early_01
```

输出时域/频域图、振幅对照、衰减描述指标、按装夹批次的统计及early_signal_report.md。
检查两类频率和衰减是否有区别；检查差异是否只由敲击力度造成；检查重新装夹是否破坏差异。
程序提供量化依据，最终“是否值得继续训练”和补采结论需要罗检查真实数据后写入报告。
没有数据时报告PENDING，不生成虚假的可分性结论。

## 两分类训练和独立测试（10月9日）

```powershell
.\.venv\Scripts\python.exe -m cnn_analysis train --config cnn_analysis/config_real.json --data pc_collector/collected_data --manifest cnn_analysis/data/manifest_real.csv --out cnn_analysis/runs/two_class_01
```

默认配置使用 `split_strategy: "stratified_hit"`，对应固定木板、相同采集条件下的识别实验。
每类独立敲击按约60%训练、20%验证、20%测试分层划分，不要求更换木板或凑装夹批次。
同一次敲击的JSON/JSONL副本先去重；完整敲击只属于一个分区，增强或切片必须在划分之后进行且留在原分区。
每类至少3次才能划分，但这是程序下限，不是有效评估所需数量；建议每类先采100～200次。
这种结果只评价已知木板、相似采集条件下的新敲击，不说明能识别新木板或重新装夹后的数据。

若需要评估装夹变化，在配置中改为 `split_strategy: "group"`。
该模式保留原来的全局 `(experiment_batch,reclamp_batch)` 隔离，至少5个真实独立实验/装夹组；同组跨木棍、跨位置也不拆开。
两种策略均要求每个分区含两类，验证集选择最优轮次，测试集不参与选模型。
不要为凑分区虚构批次。split_groups.json在敲击划分模式保存sample_id，在分组模式保存E/R组名；evaluation.json记录策略。

对照包括：简单信号特征分类器、只看振幅的分类器、去除振幅后的形状分类器。
它们用于检查CNN是否有优势、区别是否主要来自振幅，不会自动证明因果或泛化。
特征标准化只用训练集拟合。信号分析中的FFT仅作诊断和简单分类器对照，不改变CNN的时域输入。

数据不足、类别缺失或来源未知时，生成training_status.json记录原因，预留后续入口，不导出正式模型。

## 输出怎么看

| 文件 | 作用 |
|---|---|
| data_audit.json | 接收、拒收原因、重复副本、来源/标签缺失 |
| model_inputs.npz | 时域inputs、频谱spectrum_inputs，与编号、批次 |
| candidate_comparison.json | 两套CNN输入约定、状态和对照结果；默认不选定最终方案 |
| signal_analysis.png / signal_features.csv | 波形、频域、振幅、衰减指标 |
| signal_report.json / early_signal_report.md | 信号分析统计与待完成结论 |
| split_groups.json / split_manifest.csv | 核对敲击或批次隔离和每条样本所属集合 |
| training_curves.png / history.json | 学习曲线、最优轮次依据 |
| confusion_matrix.png / evaluation.json | 独立测试、错误分布、对照与子组结果 |
| error_samples.json | 需检查的错误样本定位与分数 |
| model.json / model_weights.npz | 电脑复现的模型、float32权重 |
| model_weights.h / model_card.md | 王可接入的C++参考实现与操作约定 |
| fixed_test_vectors.json | 最多10组原始信号、预处理和电脑预期输出 |
| deployment_budget.json | 参数、MAC和缓冲内存估算 |

真实可分性和模型效果不能只看训练准确率。优先看与所选策略对应的test、各位置/批次/释放方式分组结果以及错误波形。
每组数目太少、不同类释放方式不一致或电气质量未确认时，不能给出可靠验收结论。

## 单次预测

输入应是王保存的单次JSON，不需要训练标签。

```powershell
.\.venv\Scripts\python.exe -m cnn_analysis predict --model cnn_analysis/runs/two_class_01/time/model.json --record pc_collector/collected_data/samples/实际样本编号.json
```

异常返回INVALID_SIGNAL；分数并列或低于已配置阈值返回LOW_CONFIDENCE。演示模型始终标记DEMO_ONLY。
原9月28日四类output_interface.py保留；新增两类输出版本V0_2，不能混用。
频谱预测将--model改为spectrum/model.json，predict自动读取该模型的输入约定并执行频谱变换。
上述训练输出文件（模型、曲线、混淆矩阵、固定测试信号等）分别保存在time/和spectrum/目录；数据审核和早期信号分析位于共同输出目录。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s cnn_analysis/tests -v
```

导出的C++参考未在ESP32执行验证。使用fixed_test_vectors核对电脑和固件；真机内存、时延、一致性属于后续集成。
本阶段没有改采集固件、电路或OLED，不包含10月11日四分类升级。

若本机已有C++编译器，可对每套导出包执行电脑端C++核对：

```powershell
.\.venv\Scripts\python.exe -m cnn_analysis.verify_export cnn_analysis/runs/demo_01/time --compiler c++
.\.venv\Scripts\python.exe -m cnn_analysis.verify_export cnn_analysis/runs/demo_01/spectrum --compiler c++
```

该检查实际编译头文件、运行10组信号，对照预处理与分数，结果写入cpp_checks/verification.json。
不代表已经在ESP32运行。频谱头文件是直接DFT参考；真机部署需优化FFT并核对，不能直接假设满足实时性。
