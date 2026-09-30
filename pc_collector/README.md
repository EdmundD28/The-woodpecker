# 电脑采集工具

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
