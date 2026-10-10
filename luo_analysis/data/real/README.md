# 实测数据入口

把固件A采集的JSONL或单次JSON放到这里，或通过命令行直接指定外部目录。
仓库没有提供可确认来源、带类别和多个装夹批次的实测数据，因此这里没有虚构文件。

复制 `../manifest_template.csv`，逐条填写实际来源与释放方式；类别也可在配置中填写经确认的stick_labels。
source必须是real或synthetic，不能根据文件名、WOOD编号或波形自动猜测。
正式训练要求所有接收数据source=real。不同实验不得重复使用相同sample_id。
