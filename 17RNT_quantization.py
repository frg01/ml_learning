'''
量化是通过降低参数精度，优化推理行为
量化主要针对 前馈层（Feed-Forward） 和 注意力中的投影层（Attention Projection Layers） 进行
意味着对线性层（Linear Layers） 进行量化，因为前馈层和投影层本质上都是线性变换。

| 特征                  | 对称量化（Symmetric）             | 非对称量化（Asymmetric）      |
| ------------------- | --------------------------- | --------------------------------- |
| 公式                  | $x_q = \text{round}(x / s)$ | $x_q = \text{round}((x - z) / s)$ |
| 偏移项 $z$（zero-point） | 一般为 0                   | 需要非 0 的 zero-point              |
| 量化区间                | 通常为 [-127, 127]        | 通常为 [0, 255]                    |
| 量化参数                | 只有缩放因子 $s$             | 缩放因子 $s$ + 零点 $z$              |
| 数值对称性               | 是                          | 否                                 |
| 适用场景                | 权重（中心对称分布）           | 激活值（偏移正向）                   |


对称量化（权重）： 找到最小最大激活值决定范围，计算缩放因子和零点，进行量化。 精度丢失使推理速度加快。（静态、可预处理）
非对称量化（激活值）：动态量化,激活值。
'''


### 对称量化 -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.quantization

# 假设 FP32 的张量
fp32_values = torch.tensor([3.0, -5.5, 0.0, 6.0, -6.0, 2.5], dtype=torch.float32)
print(f"FP32 的示例张量: {fp32_values}\n")

# 定义 PyTorch 的量化和反量化函数
def pytorch_quantize(fp32_tensor):
    # 使用 min-max 范围计算缩放因子，指定 dtype 为 torch.qint8
    q_params = torch.quantization.MinMaxObserver(dtype=torch.qint8)
    q_params(fp32_tensor)
    scale, zero_point = q_params.calculate_qparams()

    # 量化
    int8_tensor = torch.quantize_per_tensor(fp32_tensor, scale.item(), zero_point.item(), dtype=torch.qint8)
    return int8_tensor, scale.item(), zero_point

def pytorch_dequantize(int8_tensor):
    # 反量化
    fp32_tensor = int8_tensor.dequantize()
    return fp32_tensor

# 量化并获取 PyTorch 结果
int8_tensor, scale, zero_point = pytorch_quantize(fp32_values)
print("PyTorch 量化后的 INT8 数值: ", int8_tensor.int_repr())
print("PyTorch 使用的 scale:", scale, "zero_point:", zero_point)

# 反量化
recovered_fp32_pytorch = pytorch_dequantize(int8_tensor)
print("PyTorch 反量化恢复后的 FP32 数值: ", recovered_fp32_pytorch)

print("\n=====================\n")

# 对比与自定义的量化方式
def custom_quantize_compare(fp32_values):
    # 获取张量数值的最小值和最大值
    x_min, x_max = fp32_values.min().item(), fp32_values.max().item()
    
    # 定义量化后整数数值的范围
    qmin, qmax = -128, 127  # 对应 torch.qint8
    
    # 计算 scale
    scale_custom = (x_max - x_min) / (qmax - qmin)
    
    # 非对称量化
    initial_zero_point = qmin - x_min / scale_custom
    zero_point_custom = int(round(initial_zero_point))
    
    # 将 zero_point 限制在 [qmin, qmax] 范围内
    zero_point_custom = max(qmin, min(qmax, zero_point_custom))
    
    print("自定义计算的 scale:", scale_custom, "zero_point:", zero_point_custom)
    
    def quantize(fp32_tensor, scale, zero_point):
        # 计算量化值
        int8_tensor = torch.round(fp32_tensor / scale) + zero_point
        # 限制在 [qmin, qmax] 范围内
        int8_tensor = torch.clamp(int8_tensor, qmin, qmax)
        return int8_tensor.to(torch.int8)
    
    def dequantize(int8_tensor, scale, zero_point):
        # 反量化
        fp32_tensor = (int8_tensor.float() - zero_point) * scale
        return fp32_tensor
    
    # 量化
    int8_values_custom = quantize(fp32_values, scale_custom, zero_point_custom)
    print("自定义对称量化后的 INT8 数值: ", int8_values_custom)
    
    # 反量化
    recovered_fp32_custom = dequantize(int8_values_custom, scale_custom, zero_point_custom)
    print("自定义对称反量化恢复后的 FP32 数值: ", recovered_fp32_custom)

# 运行自定义量化并比较
custom_quantize_compare(fp32_values)

print("\n=====================\n")

# 使用 fp32_values 作为线性层参数

# 定义一个简单的线性模型
class SimpleLinearModel(nn.Module):
    def __init__(self, weights, bias=None):
        super(SimpleLinearModel, self).__init__()
        # 假设输入特征数为6，输出特征数为1，用于匹配之前定义的张量，你也可以试试 (1, 6)，记得对应修改 weights.view(6, 1)
        self.linear = nn.Linear(6, 1, bias=False)
        # 初始化权重
        self.linear.weight = nn.Parameter(weights.view(1, 6))  # 权重形状为 [out_features, in_features]
    
    def forward(self, x):
        return self.linear(x)

# 创建 FP32 模型
fp32_weights = fp32_values  # [6]
model_fp32 = SimpleLinearModel(fp32_weights)

# 打印 FP32 模型的权重
print("FP32 模型的权重:\n", model_fp32.linear.weight)

# 使用默认量化配置
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 准备量化
torch.quantization.prepare(model_fp32, inplace=True)

# 量化权重
torch.quantization.convert(model_fp32, inplace=True)

# 打印量化后的权重
print("量化后的 INT8 模型的权重:\n", model_fp32.linear.weight())

# 获取量化参数
weight_observer = model_fp32.linear.weight().q_per_channel_scales()
weight_zero_points = model_fp32.linear.weight().q_per_channel_zero_points()
print("量化权重的 scale:", weight_observer)
print("量化权重的 zero_point:", weight_zero_points)


### 非对称量化  -------------------------------------------------------------------------------------
model_fp32 = SimpleLinearModel(fp32_weights)

# 打印 FP32 模型的权重
print("FP32 模型的权重:\n", model_fp32.linear.weight)


import torch.quantization as quant

# 自定义的 qconfig，使用非对称量化 per_tensor_affine代表非对称
custom_qconfig = quant.QConfig(
    activation=quant.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=quant.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)

# 应用自定义的 qconfig 到模型
model_fp32.qconfig = custom_qconfig

# 插入量化准备步骤
quant.prepare(model_fp32, inplace=True)

# 量化权重
quant.convert(model_fp32, inplace=True)

# 打印量化后的权重
quantized_weight = model_fp32.linear.weight()
print("量化后的 INT8 模型的权重（int_repr）: \n", quantized_weight.int_repr())
print("量化权重的 scale:", quantized_weight.q_scale())
print("量化权重的 zero_point:", quantized_weight.q_zero_point())

