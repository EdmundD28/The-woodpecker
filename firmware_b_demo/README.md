# 固件 B：OLED 占位展示版

独立工程，上传B会替换板上A；A源码未修改。所有分类是TEST_ONLY，不是真实木材识别。

## 操作

- OLED SSD1306 128×64：VCC→3V3、GND→GND、SDA→21、SCL/SCK→22，地址0x3C。
- 压电经过保护和1.65V偏置电路后接GPIO34，不可裸接。
- VS Code打开此目录，PlatformIO Build；连接板子后Upload；串口115200。
- 开机初始化后自动Ready；每次等Ready再敲，无需按钮。
- 收齐1024点后质量检查、预处理、占位分类；显示有效次数及本次类别1–4。
- 三次有效敲击多数表决，无多数显示Uncertain；最终结果保留5秒后自动新组。
- CLIPPED（削顶）、SAMPLE_LOSS（丢样）、WEAK_SIGNAL（弱信号）不计数，错误提示1.5秒后重试。
- 串口C/c软件触发真实ADC窗口；静止信号通常被拒为WEAK_SIGNAL，不是合成数据命令。
- OLED未接/地址错误时串口FATAL并停止；缺屏不能显示屏幕错误。

## 模型接口

依据lhf分支pc_collector/preprocess_batch.py：16kHz、1024点，减窗口均值再除2048，float32连续数组对应[1024,1]，TIME_V0_1，不做FFT。
preprocess()和classifyPlaceholder()独立，后续替换真实模型。
output_interface.py内部class_id=0..3，OLED显示+1；木材种类映射仍待罗确认。
占位分类用绝对峰值四档（0.20/0.40/0.60），仅验证链路，不读取木棍编号作预测、不伪造置信度。
model_interface_v0_1.md目前为空，未发现可部署模型。正式模型分数、量化输入和低置信度阈值待提供。

## 采样边界与验收

沿用已验证A版I²S ADC+DMA：GPIO34、16kHz、1024点、触发前256点、阈值300ADC单位。
采样期间不刷新OLED；收齐后停止ADC处理/显示，恢复时清旧DMA并补前缓存。结果/错误提示期间不接收敲击，恢复有500ms重触发保护。
B不输出A的完整JSONL，不负责训练数据保存。A20次稳定性和罗读取真实A数据应独立验收。

开机及恢复先显示Preparing，保护期结束才显示Ready；Ready刷新后约16ms用于重新填充前缓存，操作时建议看见Ready后再从容敲击。
OLED引脚集中在main.cpp的OLED_SDA、OLED_SCL、OLED_ADDRESS常量。DMA清旧数据限制为8次读取，避免无限循环；采集中250ms仍不完整会拒为SAMPLE_LOSS。

## 无硬件逻辑测试

test/host/test_firmware.cpp直接包含实际main.cpp，只模拟Arduino、OLED、I²C及DMA接口。不验证物理电压、采样率、OLED像素布局或真实通信。
可使用本机C++17编译器：`c++ -std=c++17 -I test/host test/host/test_firmware.cpp -o test/host/test_firmware.exe`，再运行生成程序。
覆盖四类演示返回、预处理、三次投票/无多数、异常不计数、触发前缓存与1024点窗口、软件触发、超时、计时回绕及自动清零。硬件驱动返回值由模拟器提供，需真机另验。

真机检查：开机Ready无FATAL；三次有效敲击依次1/3、2/3、3/3；最终结果5秒后清零；异常不加次数；三类各一票显示Uncertain；内部0..3与屏幕1..4对应。
依赖由PlatformIO安装Adafruit SSD1306及GFX。编译通过不代表真机OLED或采样率已验证。
