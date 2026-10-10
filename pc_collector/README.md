# 电脑采集工具

## 启动后逐项设置批量采集

在VS Code选择“运行Python文件”运行`collector.py`，或执行`py collector.py`。
程序依次询问COM口、木头编号、实验批次、重新装夹批次、位置、采集次数、起始敲击编号和是否开启监控。
也可以运行`py collector.py --interactive`强制进入问答。

确认后人工敲击，程序按阈值触发，每次保存一条并自动递增H编号；达到指定次数后停止读取。
本批木头、实验批次、装夹批次和位置保持不变。异常记录保留原标记并计入次数，不能直接当作合格训练数据。
实验信息由电脑端按输入填写，固件负责原始信号与异常检测，无需为更换木头重新上传固件。
最终记录仍按`WOOD_IMPACT_V1`验证，字段结构未改变。

续采同一组实验时填写接着上次的起始H编号，避免跨文件出现重复样本编号。
每次会话有独立JSONL和PNG目录，按Ctrl+C中止也保留已经写入的数据。
保存位置固定在本工具目录的`collected_data`，不依赖启动终端的位置。

## 安装依赖

在本目录打开PowerShell：

```powershell
py -m pip install -r requirements.txt
```

## 真实敲击自动触发

关闭PlatformIO Serial Monitor，然后使用项目默认端口COM7运行：

```powershell
py collector.py --port COM7 --impact-point 1 --trigger auto --count 1 --show
```

如COM7的编号变化，可把`--port COM7`改为`--port auto`。交互终端中省略`--port`也会自动识别；通过管道接入模拟器时，省略`--port`仍从标准输入读取。

程序会自动发送位置编号`1`。看到“请敲击木头”后敲击位置1，收到有效记录后将：

- 保存JSONL到`collected_data`；
- 保存PNG到`collected_data/plots`；
- 弹出Matplotlib波形窗口。

位置2或3分别修改为：

```powershell
--impact-point 2
--impact-point 3
```

连续采集五次位置1：

```powershell
py collector.py --port COM7 --impact-point 1 --trigger auto --count 5
```

连续采集并实时刷新同一个波形窗口：

```powershell
py collector.py --port COM7 --impact-point 1 --trigger auto --count 5 --live
```

`--live`会在采集开始时打开一个窗口。ESP32每完成一次1024点采样并输出JSON，窗口立即更新为最新一次敲击；采集结束后保留最后一条波形，关闭窗口即可结束。由于当前协议按完整JSON发送，它显示的是“每次敲击完成后的实时刷新”，不是64 ms采样过程中逐点滚动。

## 连续监控压电变化

循环显示最近5秒的连续预览，并在下半图显示最近一次完整敲击：

```powershell
py collector.py --port COM7 --impact-point 1 --trigger auto --monitor
```

`--monitor`会让ESP32保持16 kHz内部采样，并把每64个原始点中偏离基线最大的点作为一个预览点发送，即每4 ms更新一次、约250 Hz。这样比简单抽取单点更不容易漏掉高频敲击。上半图循环滚动显示最近5秒，用于观察基线、噪声和敲击是否发生；下半图显示最近一次完整的16 kHz、1024点（64 ms）波形。监测会一直运行，直到关闭图窗或按Ctrl+C；`--count`不限制监测时长。只有完整记录写入训练JSONL，预览行不会保存为训练数据。

没有连接ESP32时，可以用模拟器查看相同界面：

```powershell
py device_simulator.py --repeats 1 --preview | py collector.py --monitor
```

模拟器会为位置1、2、3各生成两秒预览，并在每段预览中间模拟一次敲击。上图会按真实时间滚动，下图在每条完整JSON到达后更新。最后一幅图会保持显示，关闭图窗结束。

自动测试时如果不想等待真实时间，可使用：

```powershell
py device_simulator.py --repeats 1 --preview --preview-fast | py collector.py --no-plot
```

## 软件触发

```powershell
py collector.py --port COM7 --impact-point 1 --trigger software --count 1 --show
```

程序会自动发送`1`和`C`，不需要打开Serial Monitor。

## 模拟器兼容模式

原有管道方式仍然可用：

```powershell
py device_simulator.py --repeats 1 | py collector.py
```

## 注意事项

- Serial Monitor和collector不能同时打开同一个COM口。
- 打开串口通常会让ESP32复位，因此collector等待2秒后才发送命令。
- `--show`会弹出波形窗口；关闭窗口后程序才会结束或继续下一条。
- `--live`只使用一个窗口连续刷新，适合多次真实敲击采集。
- `--monitor`循环显示最近5秒连续预览和最近一次64 ms完整波形，直到关闭窗口或按Ctrl+C。
- `--show`和`--live`不能同时使用。
- `--monitor`不能与`--show`、`--live`或`--no-plot`同时使用。
- 不需要弹窗时省略`--show`，PNG仍会自动保存。
- 使用`--no-plot`可以只保存JSONL。
# 单次敲击文件（新增）

交互批量采集JSONL现在按“启动时间_木头编号_敲击位置_计划采样次数”命名，例如 `20261008_1430_WOOD01_P02_N020.jsonl`。时间精确到分钟；同一分钟同参数重名时追加 `_02`、`_03` 等序号，独占创建不覆盖旧文件。N020表示计划保存20条（不是1024个采样点），中途停止可能不足20条，以实际行数为准。
非交互模式若未提供木头信息，名称使用UNKNOWN，不猜测木头编号；不限条数的监控/管道输入使用Nstream。正式实验建议使用交互设置以获得准确批次文件名。

每条记录独立保存到 `collected_data/samples/<sample_id>.json`，例如 `WOOD01-E01-R00-P01-H001.json`。
WOOD01是木头编号，E01实验批次，R00重新装夹批次，P01敲击位置，H001敲击编号。文件内sample_id与文件名完全一致。
同时保留原session JSONL供现有训练程序批量读取；不要把JSON和JSONL同时当成不同样本导入，以免重复。
PNG仍保存在 `collected_data/plots/<session名称>/<sample_id>.png`，与单次JSON同名。关闭绘图或绘图失败时仅有数据文件。
同编号JSON已存在时停止采集，绝不覆盖；续采需填写下一个敲击编号，或改成实际的新实验批次。旧文件不会被自动拆分或修改。
