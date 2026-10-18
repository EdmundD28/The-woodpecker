# 信号分析、神经网络训练与质量检查

## 当前采用的频谱模型

默认部署模型为`REAL-RMSLOG-fb6b2bf9d8ef`，文件为`../firmware_b_demo/include/models/spectrum_model.json`。它采用逐击均方根归一化＋对数压缩，三层2/2/2步长卷积与展平头不变，共12114参数。

`triplet.predict`自动读取新模型中的`WOOD_TRIPLET_RMS_LOG_V1`输入约定及拟合参数。新预处理不能与旧权重混用；端点统计共享，中点单独拟合。模型JSON中的来源路径只保留文件名，训练文件散列保留。

本批重训入口为`py -m luo_analysis.adopt_rms_log --out luo_analysis/runs/新的采用版目录`。该入口需要本机未发布的既有跨场地实验目录与采集数据，只生成模型，不自动覆盖固件。旧`triplet train`及单点入口继续用于历史对照。

部署版将桌面与地毯共同训练，174/56/65条敲击分为训练/验证/开发留出；归一化和对数参数只拟合训练池，5条指定异常排除。630个开发留出组合及16个不复用组均判对，但不是独立新场地验收。完整方法比较及采用边界见[模型说明](../firmware_b_demo/include/models/README.md)。


从仓库根目录运行命令。无需连接ESP32，采集工作仍由王的pc_collector负责。

## 单次敲击质量：只学习正常范围

`quality.py`将用户指定的5条异常（木头2点1的H009/H027/H038、点2的H036/H037）固定留作异常测试。
其他记录按木头和位置分层划分正常训练、正常阈值校准、正常留出测试，比例约60%/20%/20%。
这不覆盖或修改历史识别模型、训练产物或原始采集文件；实验目录分别保存四个互不重叠的数据文件与划分清单。

```powershell
py -m luo_analysis.quality train --data 采集文件1.jsonl 采集文件2.jsonl 其他采集文件.jsonl --out luo_analysis/runs/新的质量实验目录
py -m luo_analysis.quality predict --model 实验目录/quality_model.json --record 单次敲击.json
```

模型存储正常参考特征，比较新波形与最近5条参考的距离。位置1和3共用端点池，位置2使用中点池；每池均覆盖两根木头，预测不需要木头编号。
训练侧拟合稳健标准化；阈值只由正常校准集按目标5%误拦率的有限样本分位数确定。5条指定异常不用于拟合、选模型或选阈值。
既有丢样/削顶/弱信号检查与形态距离检测使用同一个预测入口，返回`retest`与原因。
本实验是已观察这5条波形后的回顾性检验，其余295条暂按正常，未经逐条人工确认。查看report.md、summary.json和预测清单，不将小测试集结果解释为现场保证。
质量检查已通过 `export_quality.py` 导出并接入固件 B；部署行为及限制见[展示固件说明](../firmware_b_demo/README.md)。

## 固定两根木头：三点联合判断

`triplet.py`复用同一三层CNN，将输入改成三个位置通道，直接输出WOOD01或WOOD02，不输出实心/空心。
每组必须包含点1、点2、点3各一条。允许123或321；中点保持在中间，两端互换。
正反方向均参与训练；推理平均两个方向的分数，交换两端后输出完全一致。
这不表示已验证实际翻面或重新装夹导致的新波形变化。

从仓库根目录运行三点训练：

```powershell
py -m luo_analysis.triplet train --data 位置1文件.jsonl 位置2文件.jsonl 位置3文件.jsonl 第二根位置1文件.jsonl 第二根位置2文件.jsonl 第二根位置3文件.jsonl --out luo_analysis/runs/新的三点实验目录
```

按相同H编号形成基础三点组，再按木头编号分层划分，原始敲击不会跨训练、验证、测试。
训练池内部额外随机配组；验证和测试各自穷举三个位置的组合。配组不创造新的独立实测记录。
两套模型按验证组合准确率、再按验证损失选定；选定后才计算留出测试。
完整结果保存在report.md、summary.json、split_manifest.csv；两套权重及C++导出分别位于time和spectrum目录。

三条新记录的预测入口：

```powershell
py -m luo_analysis.triplet predict --model 实验目录/spectrum/model.json --records 第一个端点.json 中点.json 另一个端点.json
```

预测只读取波形，不以记录内的木头编号决定类别。123和321有效，缺点、重复点和其他中点位置会拒收。
三点C++入口为`wood_triplet_time`或`wood_triplet_spectrum`中的`preprocess_three`及`infer_either_direction`；缓冲预算见deployment_budget.json。
可用`py -m luo_analysis.verify_export 实验目录/spectrum --compiler c++`编译核对导出，仍不代表ESP32实机已验证。

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
py -m luo_analysis prepare --data ../The-woodpecker/pc_collector/collected_data/20261017_1737_WOOD01_P01_N020.jsonl --out luo_analysis/runs/current_capture_01
```

读取不需要猜测类别。正式训练仍需确认两类标签和真实来源；只有一类时不生成两分类模型。

需要Python 3.10+；Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r luo_analysis/requirements.txt
.\.venv\Scripts\python.exe -m luo_analysis demo --out luo_analysis/runs/demo_01
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
.\.venv\Scripts\python.exe -m luo_analysis prepare --data luo_analysis/runs/demo_01/demo_data/demo.jsonl --out luo_analysis/runs/interface_01
```

上述模拟数据仅用于检查读取。pc_collector/examples里的旧16kHz、1024点示例与当前默认配置不匹配，会被拒收；不能作为当前实测数据。
实际文件直接传给--data，或放到luo_analysis/data/real目录。
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
.\.venv\Scripts\python.exe -m luo_analysis analyze --config luo_analysis/config_real.json --data pc_collector/collected_data --manifest luo_analysis/data/manifest_real.csv --out luo_analysis/runs/early_01
```

输出时域/频域图、振幅对照、衰减描述指标、按装夹批次的统计及early_signal_report.md。
检查两类频率和衰减是否有区别；检查差异是否只由敲击力度造成；检查重新装夹是否破坏差异。
程序提供量化依据，最终“是否值得继续训练”和补采结论需要罗检查真实数据后写入报告。
没有数据时报告PENDING，不生成虚假的可分性结论。

## 两分类训练和独立测试（10月9日）

```powershell
.\.venv\Scripts\python.exe -m luo_analysis train --config luo_analysis/config_real.json --data pc_collector/collected_data --manifest luo_analysis/data/manifest_real.csv --out luo_analysis/runs/two_class_01
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
.\.venv\Scripts\python.exe -m luo_analysis predict --model luo_analysis/runs/two_class_01/time/model.json --record pc_collector/collected_data/samples/实际样本编号.json
```

异常返回INVALID_SIGNAL；分数并列或低于已配置阈值返回LOW_CONFIDENCE。演示模型始终标记DEMO_ONLY。
原9月28日四类output_interface.py保留；新增两类输出版本V0_2，不能混用。
频谱预测将--model改为spectrum/model.json，predict自动读取该模型的输入约定并执行频谱变换。
上述训练输出文件（模型、曲线、混淆矩阵、固定测试信号等）分别保存在time/和spectrum/目录；数据审核和早期信号分析位于共同输出目录。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s luo_analysis/tests -v
```

导出的C++参考未在ESP32执行验证。使用fixed_test_vectors核对电脑和固件；真机内存、时延、一致性属于后续集成。
本阶段没有改采集固件、电路或OLED，不包含10月11日四分类升级。

若本机已有C++编译器，可对每套导出包执行电脑端C++核对：

```powershell
.\.venv\Scripts\python.exe -m luo_analysis.verify_export luo_analysis/runs/demo_01/time --compiler c++
.\.venv\Scripts\python.exe -m luo_analysis.verify_export luo_analysis/runs/demo_01/spectrum --compiler c++
```

该检查实际编译头文件、运行10组信号，对照预处理与分数，结果写入cpp_checks/verification.json。
不代表已经在ESP32运行。频谱头文件是直接DFT参考；真机部署需优化FFT并核对，不能直接假设满足实时性。

## 2026年10月19日：桌面／地毯双向迁移

`cross_surface.py`单独重训实验模型，分别只用地毯或桌面训练；目标支撑面不参与训练、校准或选择轮次。
5条指定异常只做测试；每个组合是一根木头、同一支撑面、三个位置各一条，推理支持123和321。
三组固定划分比较时域／频谱网络、三档质量检查、幅值／形态对照，以及不重复使用原始记录的三点测试。
不自动覆盖固件模型。场地标签按用户说明映射，原始文件散列依赖`20261019_surface_labels_audit`审核结果。

```powershell
& '../The-woodpecker/.venv/Scripts/python.exe' -m luo_analysis.cross_surface --out luo_analysis/runs/新的实验目录
& '../The-woodpecker/.venv/Scripts/python.exe' -m luo_analysis.cross_surface_report --run luo_analysis/runs/新的实验目录
& '../The-woodpecker/.venv/Scripts/python.exe' -m luo_analysis.cross_surface_verify --run luo_analysis/runs/新的实验目录
```

本轮结果：`luo_analysis/runs/20261019_cross_surface_v1/report.md`。
`summary.json`含全部指标；`verification.json`记录重新加载权重、重建源场地质量参考与阈值、划分隔离及预测复核。
随机组合之间共享原始记录，组合数不能当成独立采集量；木头1点1只有5条地毯记录。
宽松质量档误拦较少不代表识别已经稳定跨场地，完整结论和逐条待复核波形见报告。

## 2026年10月19日：频谱卷积步长对照

`stride_study.py`复用上一轮双向迁移的全部原始划分和三点组合，对照步长2/2/2、2/2/1、1/1/1及1/1/1加151频段汇总。
最后一项在所有卷积之后压缩频率轴，参数量与原结构相同，用来检查提前降采样是否存在代价。
三个种子、两个方向，共24个模型；全部权重及源场地选型冻结后才计算目标成绩。
改变步长也改变感受野，直接展平还增加参数，因此不能把所有变化都解释成频率细节的作用。

```powershell
& '../The-woodpecker/.venv/Scripts/python.exe' -m luo_analysis.stride_study --out luo_analysis/runs/新的步长实验目录
& '../The-woodpecker/.venv/Scripts/python.exe' -m luo_analysis.stride_study_verify --run luo_analysis/runs/新的步长实验目录
```

本轮目录：`luo_analysis/runs/20261019_stride_ablation_v1/`。`protocol_locked.json`为固定实验规则；各划分的`selection_locked.json`记录仅按源场地验证损失选型；`all_models_frozen.json`记录目标测试前全部权重散列。
完成后`report.md`报告每种结构、每个种子、两根木头的结果，`verification.json`记录权重重载、预测重算、端点交换一致性和原结构复现核验。实验不替换固件。
本轮结果：减少或取消步长降采样没有稳定改善双向迁移。完全无降采样的平均平衡准确率在两个方向均下降；等参数晚压缩在地毯→桌面仅提高1.22个百分点且只有一个种子改善，不复用组均值与原结构相同；桌面→地毯下降0.82个百分点。未替换固件。当前环境与前轮NumPy版本不同，比较使用本轮同环境重训基线；旧环境单组复现的训练历史及权重与前轮完全一致，详见实验报告。
