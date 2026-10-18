# The-woodpecker

ESP32 木材敲击展示系统。`main` 以本次固件 B 和神经模型交付为准：依次采集端点、中点、另一端点，每点通过质量检查后，由三路卷积神经网络联合识别 `WOOD01` 或 `WOOD02`，并在 OLED 显示结果。

模型只识别当前两根已知木头，不能据此判断木材品种、实心空心或缺陷。已完成电脑端验证和固件编译；本次未烧录，实机采样时序、内存峰值及新场地效果仍需测量。

## 分支约定

- `main`：固件 B、部署模型、训练与分析源码、测试和交付说明。
- `wood-impact-system-wjy`（简称 `wjy`）：固件 A 与实验采集流程的开发改动。
- 本次按目录整理提交，没有合并成员分支；固件 A 的源码改动不纳入本次提交。

## 交付内容

| 路径 | 内容 |
|---|---|
| [firmware_b_demo](firmware_b_demo/README.md) | 展示固件、接线、烧录、操作和质量档位 |
| [部署模型](firmware_b_demo/include/models/manifest.json) | 默认时域模型、可选频谱模型及来源散列 |
| [验证记录](firmware_b_demo/VERIFICATION.md) | 编译、数值对照、实机验证边界 |
| [luo_analysis](luo_analysis/README.md) | 训练、导出、质量检查、跨支撑面实验及测试 |
| [协议校验](pc_collector/protocol.py) | 模型读取数据所需的固定协议校验模块 |
| [协议示例](pc_collector/examples/protocol.md) | 协议说明与旧采样规格的回归测试示例 |
| [固定测试材料](firmware_b_demo/test/fixtures/README.md) | 部署模型的数值对照输入 |

训练输出、完整实验目录、原始采集目录、构建产物及缓存不再进入版本跟踪。固定测试材料作为复现依赖单独提交。历史生成文件仅从版本跟踪移除，本机文件保留。

## 快速验证

需要 Python 3.10+、PlatformIO，以及电脑端数值核对所用的 C++ 编译器。

```powershell
py -m venv .venv
./.venv/Scripts/python.exe -m pip install -r luo_analysis/requirements.txt
./.venv/Scripts/python.exe -m unittest discover -s luo_analysis/tests -v
pio run -d firmware_b_demo -e esp32dev -e esp32dev_spectrum
./firmware_b_demo/run_host_tests.ps1 -Compiler 'C:/工具目录/zig.exe'
```

`-Compiler` 也支持兼容命令行的 `g++` 或 `clang++` 可执行文件路径。部署权重和固定测试输入已包含在仓库，无需相邻成员分支或本机训练输出即可执行电脑端固件核对。跨支撑面等研究脚本另需各自的实验数据与输出，详见模型目录说明。

默认构建为时域模型与场地容错质量档；烧录、触摸操作、串口命令见固件 B 说明。
