# # 训练时的显存占用
# # 主要由以下部分组成：
# # 模型参数：模型的权重和偏置。
# # 优化器状态：如动量和二阶矩估计等，取决于优化器的类型，单纯的 SGD 不占显存。
# # Adam 优化器：需要存储一阶和二阶动量，各自占用模型参数大小的空间。
# # 梯度：每个参数对应一个梯度，数据类型与模型参数相同。
# # 中间激活值：前向传播过程中产生的激活值，需要在反向传播中使用，其显存占用与 Batch Size、序列长度以及模型架构相关。
# # 批量大小（Batch Size）：一次处理的数据样本数量。
# # 其他开销：CUDA 上下文、显存碎片等。

# # 推理时的显存占用
# # 推理时的显存占用主要包括：
# # 模型参数：同训练阶段一致。
# # 中间激活值：仅保留当前步的激活值，相较于训练阶段，小非常多。
# # 批量大小（Batch Size）：一次处理的数据样本数量。

# # 高精度FP32: 更高的数值稳定性和模型准确性，但是显存占用更大，计算速度较慢。
# # 低精度FP16/INT8: 显存占用更小，计算速度更快，但可能会导致数值不稳定和模型准确性下降。
# # FP32：适用于小模型和数值精度较高的任务，FP16/BF16：适用于大模型，混合精度节省显存并加速计算。
# # INT8: 用于推理阶段，尤其资源有限下超大模型

# ### 混合精度torch.cuda.amp  使用FP16
# import torch
# from torch import nn, optim  # 神经网络模块， 优化器模块
# from torch.cuda.amp import GradScaler, autocast #防止精度损失的梯度缩放工具，混合精度加速训练 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = nn.Sequential(...) #定义模型
# model.to(device) 
# optimizer = optim.Adam(model.parameters(), lr=1e-3) #定义优化器
# scaler = GradScaler() #定义梯度缩放器

# for data, labels in dataloader:
#     data = data.to(device)
#     labels = labels.to(device)
#     optimizer.zero_grad()
#     with autocast(): #开启混合精度训练 默认使用FP16，示例2：autocast(dtype=torch.bfloat16)
#         outputs = model(data)
#         loss = criterion(outputs, labels) #计算损失

#     # 使用梯度缩放器进行反向传播
#     scaler.scale(loss).backward() 
#     scaler.step(optimizer) 
#     scaler.update() 

# # 推理
# model.half()
# model.to(device)
# inputs = inputs.half().to('cuda')
# outputs = model(inputs)

# ### HuggingFace Accelerate库
# from accelerate import Accelerator

# # 初始化 Accelerator，开启混合精度
# accelerator = Accelerator(fp16=True) #Accelerator(mixed_precision="bf16")
# model = ...  # 定义模型
# optimizer = ...  # 定义优化器
# dataloader = ...  # 定义数据加载器

# # 使用 accelerator.prepare() 函数包装模型、优化器和数据加载器
# model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# for data, labels in dataloader:
#     outputs = model(data)
#     loss = loss_fn(outputs, labels)

#     # 使用 accelerator 进行反向传播和优化步骤
#     accelerator.backward(loss)
#     optimizer.step()
#     optimizer.zero_grad()



# ### 不同精度下的显存占用对比
# import os,gc,torch
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# import bitsandbytes as bnb

# def load_model_and_measure_memory(precision, model_name, device):
#     if device.type != "cuda":
#             print("FP16 precision requires a CUDA device.")
#             return
#     if precision == "fp32":
#         model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#     elif precision == "fp16":
#         if device.type != "cuda":
#             print("FP16 precision requires a CUDA device.")
#             return
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True
#             ).to(device)
#     elif precision == "int8":
#         if device.type != "cuda":
#             print("FP16 precision requires a CUDA device.")
#             return
#         bnb_config = BitsAndBytesConfig(
#             load_in_8bit=True,
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             quantization_config=bnb_config,
#             device_map="auto",
#         )
#     else:
#         raise ValueError("Unsupported precision type. Use 'fp32', 'fp16', or 'int8'.")

#     # 确保所有CUDA操作完成
#     torch.cuda.synchronize()

#     mem_allocated = torch.cuda.memory_allocated(device) / 1e9
#     print(f"Precision: {precision}, Memory Allocated: {mem_allocated:.2f} GB")

#     # 清理模型和缓存
#     del model
#     gc.collect()
#     torch.cuda.empty_cache()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
# # model_name = "gpt2-large"
# model_name = "gpt2"

# for precision in ["fp32", "fp16", "int8"]:
#     print(f"\n----Loading model with {precision} precision...")
#     load_model_and_measure_memory(precision, model_name, device)

    

### 定义一个简单的线性层进行模拟，展示导入，训练和推理阶段显存的变化
import torch    
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import gc
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检查是否有可用的GPU

# 定义一个多层感知机MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(  # 按顺序连接的网络结构
            nn.Linear(28 * 28, 8192), # 输入层
            nn.ReLU(), # 激活函数， ReLU保留正值，负值归零
            nn.Linear(8192, 4096), 
            nn.ReLU(), 
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # 输出层
        )
    
    def forward(self, x): # 前向传播
        x = x.view(x.size(0), -1) # 将二维图片展平为向量
        x = self.layers(x) # 前向传播
        return x
    

# 准备数据集 （使用MNIST）
transform = transforms.Compose([
    transforms.ToTensor(), # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,)) # 归一化，对图像像素进行标准化（均值，标准差）
])

batch_size = 128

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 函数： 仅测量已分配的显存
def print_gpu_memory(stage):
    torch.cuda.empty_cache() # 清空缓存
    allocated_memory = torch.cuda.memory_allocated(device) / 1e6 # 转换为MB
    print(f"{stage} Allocated: {allocated_memory:.2f} MB") # 打印分配的显存

# 清理缓存， 确保测量准确
torch.cuda.empty_cache()
gc.collect()

# ---------------------------
# 1. 测量导入模型时的显存占用
# ---------------------------
print("Measuring memory during model import...")
print_gpu_memory('Before loading model')

model = SimpleMLP().to(device) # 多层感知机模型，转移到 CUDA 或者 CPU 上

print_gpu_memory('After loading model') # 测量显存占用

#保存导入阶段的显存占用
memory_loading = torch.cuda.memory_allocated(device) / 1e6 # 转换为MB

# ---------------------------
# 2. 测量训练过程中的显存占用，并计算数据占用的显存
# ---------------------------
print("\nMeasuring memory during training...")

criterion = nn.CrossEntropyLoss() # 多分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器
num_epochs = 1 # 训练轮数,为了快速演示

print_gpu_memory('Before training')

model.train() # 设置模型为训练模式
for epoch in range(num_epochs):
   for batch_idx, (data, target) in enumerate(train_loader):
        # 记录每个batch的显存占用
        memory_before_data = torch.cuda.memory_allocated(device) / 1e6 # 转换为MB
        data, target = data.to(device), target.to(device)

        #记录数据加载后的显存占用
        memory_after_data = torch.cuda.memory_allocated(device) / 1e6 # 转换为MB
        data_memory_usage = memory_after_data - memory_before_data # 计算数据占用的显存
        print(f"Data memory usage for batch {batch_idx+1}, Data Memory Usage: {data_memory_usage:.2f} MB")

        # 记录训练考试前的显存占用
        memory_before_training = torch.cuda.memory_allocated(device) / 1e6

        optimizer.zero_grad()            # 前向传播
        output = model(data) 
        loss = criterion(output, target) # 损失
        loss.backward()                  # 反向传播
        optimizer.step()                 # 优化

        # 记录训练后显存占用
        memory_after_training = torch.cuda.memory_allocated(device) / 1e6

        # 计算训练步骤中的额外显存占用（激活值，梯度，优化器状态等）
        training_memory_usage = memory_after_training - memory_before_training
        print(f"Training Memory Usage: {training_memory_usage:.2f} MB")

        #测量一次
        if batch_idx == 0:
            print_gpu_memory(f'Traning Epoch {epoch+1},Bathch {batch_idx+1}')
            # 保存训练阶段的显存占用
            memory_training = torch.cuda.memory_allocated(device) / 1e6
            break

# ---------------------------
# 3. 测量推理过程中的显存占用，并计算数据占用的显存
# ---------------------------
print("\nMeasuring memory during inference...")
# 清理缓存， 确保测量准确
del optimizer
del criterion

# 清理所有模型参数的梯度
for param in model.parameters():
    param.grad = None

torch.cuda.empty_cache()
gc.collect()

model.eval() # 设置模型为评估模式   
with torch.no_grad(): # 禁用梯度计算
    for batch_idx, (data, target) in enumerate(test_loader):
        # 记录每个batch的显存占用
        memory_before_data = torch.cuda.memory_allocated(device) / 1e6
        data = data.to(device)

        # 记录数据加载后的显存占用
        memory_after_data = torch.cuda.memory_allocated(device) / 1e6
        data_memory_usage = memory_after_data - memory_before_data
        print(f"Data memory usage for batch {batch_idx+1}, Data Memory Usage: {data_memory_usage:.2f} MB")

        output = model(data)

        # 测量一次

        print_gpu_memory(f'Inference Epoch {batch_idx+1}')
        # 保存推理阶段的显存占用
        memory_inference = torch.cuda.memory_allocated(device) / 1e6
        break


# ---------------------------
# 4. 整理并展示结果
# ---------------------------
print('\nSummary of GPU Memory Usage:')
print(f'Memory during model loading: {memory_loading:.2f} MB')
print(f'Memory during training: {memory_training:.2f} MB')
print(f'Memory during inference: {memory_inference:.2f} MB')

# 可视化
stages = ['Loading', 'Training', 'Inference']
memories = [memory_loading, memory_training, memory_inference]

plt.bar(stages, memories, color=['blue', 'green', 'red'])
plt.ylabel('GPU Memory Usage (MB)')
plt.title('GPU Memory Usage at Different Stages')
plt.show()