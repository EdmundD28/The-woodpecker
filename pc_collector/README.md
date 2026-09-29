# 电脑采集工具

## 安装依赖

在本目录打开PowerShell：

```powershell
py -m pip install -r requirements.txt
```

## 真实敲击自动触发

关闭PlatformIO Serial Monitor，然后运行（将COM5换成实际端口）：

```powershell
py collector.py --port COM5 --impact-point 1 --trigger auto --count 1 --show
```

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
py collector.py --port COM5 --impact-point 1 --trigger auto --count 5
```

## 软件触发

```powershell
py collector.py --port COM5 --impact-point 1 --trigger software --count 1 --show
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
- 不需要弹窗时省略`--show`，PNG仍会自动保存。
- 使用`--no-plot`可以只保存JSONL。
