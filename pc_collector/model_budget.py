# 与训练配置一致的时域模型预算。
import json
from pathlib import Path

config = json.loads((Path(__file__).resolve().parents[1]/'luo_analysis'/'config.json').read_text(encoding='utf-8'))
input_length = config['sample_count']
input_channels = config['channels']

# 每层配置：（输出通道数，卷积核长度，步长）
conv_layers = config['conv_layers']
output_shapes = []

total_params = 0
total_macs = 0

length = input_length
channels = input_channels

print("输入形状:", (length, channels))

for index, (out_channels, kernel, stride) in enumerate(
    conv_layers, start=1
):
    # SAME填充下的输出长度：向上取整
    out_length = (length + stride - 1) // stride

    # 卷积参数 = 卷积核权重 + 每个输出通道的偏置
    params = kernel * channels * out_channels + out_channels

    # MAC表示一次乘法并累加
    macs = out_length * out_channels * kernel * channels

    total_params += params
    total_macs += macs

    print(
        f"卷积层{index}: "
        f"输出=({out_length}, {out_channels}), "
        f"参数={params}, MAC={macs}"
    )

    output_shapes.append((out_length, out_channels))
    length = out_length
    channels = out_channels

# 全局平均池化没有可训练参数
print("全局平均池化输出:", channels)

# 类别数与实际训练配置一致。
class_count = len(config['classes'])
dense_params = channels * class_count + class_count
dense_macs = channels * class_count

total_params += dense_params
total_macs += dense_macs

print(f"全连接层: 参数={dense_params}, MAC={dense_macs}")

# float32的每个参数占4字节
parameter_bytes = total_params * 4

print()
print("总参数量:", total_params)
print("float32参数存储字节:", parameter_bytes)
print("float32参数存储KiB:", round(parameter_bytes / 1024, 2))
print("卷积与全连接合计MAC:", total_macs)

# 估算float32输入及中间结果的存储量
tensor_sizes = [
    ("模型输入", input_length * input_channels * 4),
    *[(f"卷积层{i+1}输出", size * channels * 4) for i,(size,channels) in enumerate(output_shapes)],
    ("全局平均池化输出", channels * 4),
    ("分类输出", class_count * 4),
]

print()
print("输入与中间结果的内存估算：")

for name, byte_count in tensor_sizes:
    print(f"{name}: {byte_count}字节，{byte_count / 1024:.2f} KiB")

# 如果每个数组都单独保留，不复用空间
all_tensor_bytes = sum(size for _, size in tensor_sizes)

# 相邻两层结果同时存在时，所需空间的最大值
largest_pair_bytes = max(
    tensor_sizes[i][1] + tensor_sizes[i + 1][1]
    for i in range(len(tensor_sizes) - 1)
)

# 原始采样值按uint16_t保存。
raw_buffer_bytes = input_length * 2

print()
print("全部上述数组保留:", round(all_tensor_bytes / 1024, 2), "KiB")
print("最大相邻数组组合:", round(largest_pair_bytes / 1024, 2), "KiB")
print("原始采样缓冲区:", raw_buffer_bytes / 1024, "KiB")
print(
    "全部上述数组加原始采样缓冲区:",
    round((all_tensor_bytes + raw_buffer_bytes) / 1024, 2),
    "KiB",
)
