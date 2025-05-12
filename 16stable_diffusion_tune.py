'''
pip install timm
pip install fairscale
pip install transformers
pip install requests
pip install accelerate
pip install diffusers
pip install einop
pip install safetensors
pip install voluptuous
pip install jax
pip install jaxlib
pip install peft
pip install deepface==0.0.92
pip install tensorflow==2.9.0  # 为了避免最后评估阶段使用deepface时的错误，这里选择降级版本
pip install keras
pip install opencv-python
'''
### 导入 ---------------------------------------------------------------
# ========== 标准库模块 ==========
import os
import math
import glob
import shutil
import subprocess

# ========== 第三方库 ==========
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# ========== 深度学习相关库 ==========
from torchvision import transforms

# Transformers (Hugging Face)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

# Diffusers (Hugging Face)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    DiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

# ========== LoRA 模型库 ==========
from peft import LoraConfig, get_peft_model, PeftModel

# ========== 面部检测库 ==========
from deepface import DeepFace

import cv2


### 设置项目路径 ------------------------------------------------------------
# 项目名称和数据集名称
project_name = "Brad"
dataset_name = "Brad"

# 根目录和主要目录
root_dir = "./"  # 当前目录
main_dir = os.path.join(root_dir, "SD")  # 主目录

# 项目目录
project_dir = os.path.join(main_dir, project_name)  # 项目目录

# 数据集和模型路径
images_folder = os.path.join(main_dir, "Datasets", dataset_name)
prompts_folder = os.path.join(main_dir, "Datasets", "prompts")
captions_folder = images_folder  # 与原始代码一致
output_folder = os.path.join(project_dir, "logs")  # 存放 model checkpoints 和 validation 的文件夹

# prompt 文件路径
validation_prompt_name = "validation_prompt.txt"
validation_prompt_path = os.path.join(prompts_folder, validation_prompt_name)

# 模型检查点路径
model_path = os.path.join(project_dir, "logs", "checkpoint-last")

# 其他路径设置
zip_file = os.path.join("./", "data/14/Datasets.zip")
inference_path = os.path.join(project_dir, "inference")  # 保存推理结果的文件夹

os.makedirs(images_folder, exist_ok=True)
os.makedirs(prompts_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(inference_path, exist_ok=True)

# 检查并解压数据集
print("📂 正在检查并解压样例数据集...")

if not os.path.exists(zip_file):
    print("❌ 未找到数据集压缩文件 Datasets.zip！")
    print("请下载数据集:\n../Demos/data/14/Datasets.zip\n并放在 ./data/14 文件夹下")
else:
    subprocess.run(f"unzip -q -o {zip_file} -d {main_dir}", shell=True)
    print(f"✅ 项目 {project_name} 已准备好！")


### 导入数据 -----------------------------------------------------------
# 训练图像的分辨率
resolution = 512

# 数据增强操作
train_transform = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),  # 调整图像大小
        transforms.CenterCrop(resolution),  # 中心裁剪图像
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 将图像转换为张量
    ]
)

### 怎么让模型理解文本 CLIPTokenizer,CLIP，全称 Contrastive Language-Image Pretraining（对比语言-图像预训练）
from transformers import CLITokenizer

# 初始化 CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# 示例 prompt
prompt_text = "A man in a graphic tee and sport coat."

# 先使用 tokenizer.tokenize 查看分词后的 token
tokens = tokenizer.tokenize(prompt_text)
print("Tokens:", tokens)

# 将文本转化为 token
inputs = tokenizer(
    prompt_text,
    padding="max_length",  # 如果输入长度不足最大长度，进行填充
    truncation=True,       # 如果输入过长，进行截断
    return_tensors="pt"    # 返回 PyTorch 张量
)

# 打印分词后的结果
print("Tokenized Input IDs:", inputs.input_ids)
print("Attention Mask:", inputs.attention_mask)


### 自定义数据集 -------------------------------------------------------------------
"""
IMAGE_EXTENSIONS：定义可接受的图像文件扩展名列表。
__init__ 方法：
    图像路径：通过遍历指定的图像文件夹，获取所有符合扩展名的图像文件路径，并排序。
    文本标注：在标注文件夹中查找所有 .txt 文件，读取其内容并存储为列表。
    一致性检查：确保图像数量与文本标注数量一致。
    文本编码：使用 tokenizer 将文本标注转换为 token IDs。
    数据转换：存储图像的预处理方法 transform。
__getitem__ 方法：
    根据索引获取图像路径和对应的文本 token ID。
    尝试加载并预处理图像，失败时返回全零张量。
__len__ 方法：返回数据集的长度。
"""
# 图片后缀， 
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

class Text2ImageDataset(torch.utils.data.Database):
    """
    (1) 目标:
        - 用于构建文本到图像模型的微调数据集
    """

    def __init__(self, images_folder, captions_folder, transform, tokenizer):
        """
        (2) 参数:
            - images_folder: str, 图像文件夹路径
            - captions_folder: str, 标注文件夹路径
            - transform: function, 将原始图像转换为 torch.Tensor
            - tokenizer: CLIPTokenizer, 将文本标注转为 word ids
        """
        # 初始化图像路径列表，并根据指定的扩展名找到所有图像文件
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
        
        self.image_paths = sorted(self.image_paths)

        # 加载对应的文本标注，依次读取每个文本文件中的内容
        caption_paths = sorted(glob.glob(os.path.join(captions_folder, "*.txt")))
        captions = []
        for p in caption_paths:
            with open(p, "r", encoding="utf-8") as f:
                captions.append(f.readline().strip())

        #确保图像和文本标注数量一致
        if len(captions) != len(self.image_paths):
            raise ValueError("图像数量与文本标注数量不一致，请检查数据集。")
        
        # 使用 tokenizer 将文本标注转换为word ids
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.input_ids = inputs.input_ids
        self.transform = transform

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        input_id = self.input_ids[idx]
        try: 
            # 加载图像并将其转换为 RGB 模式，然后应用数据增强
            image = Image.open(img_path).convert("RGB")
            tensor = self.input_ids[idx]
        except Exception as e:
            print(f"⚠️ 无法加载图像路径: {img_path}, 错误: {e}")
            # 返回一个全零的张量和空的输入 ID 以避免崩溃
            tensor = torch.zero((3, resolution, resolution))
            input_id = torch.zero_like(input_id)
        
        return tensor, input_id # 返回处理后的图像和相应的文本标注
    
    def __len__(self):
        return len(self.image_paths)
    
### 定义微调相关的函数 ------------------------------------------------------------------

def perpare_lora_model(lora_config, pretrained_model_name_or_path, model_path=None, resume=False, merge_lora=False):
    """
    (1) 目标:
        - 加载完整的 Stable Diffusion 模型，包括 LoRA 层，并根据需要合并 LoRA 权重。这包括 Tokenizer、噪声调度器、UNet、VAE 和文本编码器。

    (2) 参数:
        - lora_config: LoraConfig, LoRA 的配置对象
        - pretrained_model_name_or_path: str, Hugging Face 上的模型名称或路径
        - model_path: str, 预训练模型的路径
        - resume: bool, 是否从上一次训练中恢复
        - merge_lora: bool, 是否在推理时合并 LoRA 权重

    (3) 返回:
        - tokenizer: CLIPTokenizer
        - noise_scheduler: DDPMScheduler
        - unet: UNet2DConditionModel
        - vae: AutoencoderKL
        - text_encoder: CLIPTextModel

    加载模型组件： 依次加载了噪声调度器、Tokenizer、文本编码器（text_encoder）、VAE 和 UNet 模型。
    应用 LoRA： 使用 get_peft_model 函数将 LoRA 配置应用到 text_encoder 和 unet 模型中。这会在模型中插入可训练的 LoRA 层。
    打印可训练参数： 调用 print_trainable_parameters() 来查看 LoRA 添加了多少可训练参数。
    恢复训练： 如果设置了 resume=True，则从指定的 model_path 加载之前保存的模型权重。  
    合并 LoRA 权重： 如果 merge_lora=True，则将 LoRA 的权重合并到基础模型中，以便在推理时使用，感兴趣的话阅读：14. PEFT：在大模型中快速应用 LoRA。
    冻结 VAE 参数： 调用 vae.requires_grad_(False) 来冻结 VAE 的参数，使其在训练中不更新。
    移动模型到设备： 将所有模型组件移动到指定的设备（CPU 或 GPU），并设置数据类型。
    """
    # 加载噪声调度器，用于控制扩散模型的噪声添加和移除过程
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # 加载 Tokenizer， 用于将文本标注转换为 tokens
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    # 加载 CLIP 文本编辑器， 用于将文本标注转换为特征向量
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dytpe=weight_dtype,
        subfolder="text_encoder"
    )

    # 加载 VAE 模型
    vae = AutoencoderKL.from_trained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )

    # 加载 UNet 模型， 负责处理扩散模型中的图像生成和推理过程
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        subfolder="unet"
    )

    # 如果设置为继续训练，则加载上一次的模型权重
    if resume:
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("当 resume 设置为true时， 必须提供有效的 model_path")
        # 使用 PEFT 的from_pretrained 方法加载 LoRA 模型
        text_encoder = PeftModel.from_pretrained(text_encoder,  os.path.join(model_path, "text_encoder"))
        unet = PeftModel.from_pretrained(unet, os.path.join(model_path), "unet")

        # 确保 UNet 的可训练参数的 requires_grad 为True
        for param in unet.parameters():
            if param.requires_grad is False:
                param.requires_grad = True
        
        # 确保文本编码器的可训练参数的 requires_grad 为 True
        for param in text_encoder.parameters():
            if param.requires_grad is False:
                param.requires_grad = True

        print("f"✅ 已从 {model_path} 恢复模型权重"")

    else: 
        # 将 LoRA 配置应用到 text_encoder 和unet
        text_encoder = get_peft_model(text_encoder, lora_config)
        unet = get_peft_model(unet, lora_config)

        # 打印可训练参数数量
        print("📊 Text Encoder 可训练参数:")
        text_encoder.print_trainable_parameters()
        print("📊 UNet 可训练参数:")
        unet.print_trainable_parameters()

    if merge_lora:
        # 合并LoRA 权重到基础模型， 仅在推理时调用
        text_encoder = text_encoder.merge_and_unload()
        unet = unet.merge_and_unload()

        # 切换为评估模式
        text_encoder.eval()
        unet.eval()

    # 冻结 VAE 参数
    vae.requires_grad_(False)

    # 将模型移动到 GPU 上并设置权的数据类型
    unet.to(DEVICE, dtype=weight_dtype)
    vae.to(DEIVCE, dtype=weight_dtype)
    text_encoder.to(DEIVCE, dtype=weight_dtype)

    return tokenizer, noise_scheduler, unet, vae, text_encoder

### 准备优化器  ------------------------------------------------------------------
def prepare_optimizer(unet, text_encoder, unet_learning_rate=5e-4, text_encoder_learning_rate=1e-4):
    """
    (1) 目标:
        - 为 UNet 和文本编码器的可训练参数分别设置优化器，并指定不同的学习率。

    (2) 参数:
        - unet: UNet2DConditionModel, Hugging Face 的 UNet 模型
        - text_encoder: CLIPTextModel, Hugging Face 的文本编码器
        - unet_learning_rate: float, UNet 的学习率
        - text_encoder_learning_rate: float, 文本编码器的学习率

    (3) 返回:
        - 输出: 优化器 Optimizer
    """
    # 筛选出 UNet 中需要训练的 Lora 层参数
    unet_lora_layers = [p for p in unet.parameters() if p.requires_grad]

    # 筛选出文本编码器中需要训练的 Lora层参数
    text_encoder_lora_layers = [p for p in text_encoder.parameters() if p.requires_grad]

    # 将需要训练的参数分组并设置不同的学习率
    trainable_params = [
        {"params": unet_lora_layers, "lr": unet_learning_rate},
        {"params": text_encoder_lora_layers, "lr": text_encoder_learning_rate}
    ]

    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(trainable_params)

    return optimizer

### 定义 collate_fn 函数， 将批次数据组织成字典的形式，方便通过键名直接访问，例如 batch["pixel_values"] 和 batch["input_ids"]。
def collate_fn(examples):
    pixel_values = []
    input_ids = []
    
    for tensor, input_id in examples:
        pixel_values.append(tensor)
        input_ids.append(input_id)
    
    pixel_values = torch.stack(pixel_values, dim=0).float()
    input_ids = torch.stack(input_ids, dim=0)
    
    # 如果你喜欢列表推导式的话，使用下面的方法
    #pixel_values = torch.stack([example[0] for example in examples], dim=0).float()
    #input_ids = torch.stack([example[1] for example in examples], dim=0)
    return {"pixel_values": pixel_values, "input_ids": input_ids}


import torch
from torch.utils.data import DataLoader, Dataset

def compare_dataloaders(dataset, batch_size):
    # 第一种情况：使用自定义的 collate_fn
    train_dataloader_custom = DataLoader(
       dataset,
       shuffle=True,
       collate_fn=collate_fn,  # 使用自定义的 collate_fn
       batch_size=batch_size,
    )
    
    # 第二种情况：不使用自定义的 collate_fn（默认方式）
    train_dataloader_default = DataLoader(
       dataset,
       shuffle=True,
       batch_size=batch_size,
    )
    
    # 从每个数据加载器中取一个批次进行对比
    custom_batch = next(iter(train_dataloader_custom))
    default_batch = next(iter(train_dataloader_default))
    
    # 打印自定义 collate_fn 的输出结果
    print("使用自定义 collate_fn:")
    print("批次的类型:", type(custom_batch))
    print("批次 pixel_values 的形状:", custom_batch["pixel_values"].shape)
    print("批次 input_ids 的形状:", custom_batch["input_ids"].shape)
    
    # 打印默认 DataLoader 的输出结果
    print("\n使用默认 collate_fn:")
    print("批次的类型:", type(default_batch))
    
    pixel_values, input_ids = default_batch
    print("批次 pixel_values 的形状:", pixel_values.shape)
    print("批次 input_ids 的形状:", input_ids.shape)
    
    return custom_batch, default_batch
       
# 对比
custom_batch, default_batch = compare_dataloaders(dataset, batch_size=2)

###设置相关的参数 -------------------------------------------------------------------
#设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# For Mac M1, M2...
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

print(f"🖥 当前使用的设备: {DEVICE}")

# 模型与训练参数配置
'''
训练参数：设置批次大小、数据类型、随机种子等。
train_batch_size = 2 时，微调显存要求为 5G，在命令行输入 nvidia-smi 可以查看当前显存占用。
优化器参数：为 UNet 和文本编码器分别设置学习率。
学习率调度器：选择 cosine_with_restarts 调度器，这一点一般无关紧要。
预训练模型：指定预训练的 Stable Diffusion 模型。
LoRA 配置：设置 LoRA 的相关参数，如秩 r、lora_alpha、应用模块等。
'''

# 训练相关参数
train_batch_size = 2  # 训练批次大小，即每次训练中处理的样本数量
weight_dtype = torch.bfloat16  # 权重数据类型，使用 bfloat16 以节省内存并加快计算速度
snr_gamma = 5  # SNR 参数，用于信噪比加权损失的调节系数

# 设置随机数种子以确保可重复性
seed = 1126 # 随机种子
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Stable Diffusion LoRA 的微调参数

# 优化器参数
unet_learning_rate = 1e-4 # UNet 的学习率，控制 UNet 参数更新的步长
text_encoder_learning_rate = 1e-4  # 文本编码器的学习率，控制文本嵌入层的参数更新步长

# 学习率调度器参数
lr_scheduler_name = "cosine_with_restarts"  # 设置学习率调度器为 Cosine annealing with restarts，逐渐减少学习率并定期重启
lr_warmup_steps = 100  # 学习率预热步数，在最初的 100 步中逐渐增加学习率到最大值
max_train_steps = 2000  # 总训练步数，决定了整个训练过程的迭代次数
num_cycles = 3  # Cosine 调度器的周期数量，在训练期间会重复 3 次学习率周期性递减并重启

# 预训练的 Stable Diffusion 模型路径，用于加载模型进行微调
pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"  

# LoRA 配置
lora_config = LoraConfig(
    r=32, # LoRA 的秩，即低秩矩阵的维度，决定了参数调整的自由度
    lora_alpha=16, # 缩放系数，控制 LoRA 权重对模型的影响
    target_modules=[
        "q_proj", "v_proj", "k_proj", "out_proj",  # 指定 Text encoder 的 LoRA 应用对象（用于调整注意力机制中的投影矩阵）
        "to_k", "to_q", "to_v", "to_out.0"  # 指定 UNet 的 LoRA 应用对象（用于调整 UNet 中的注意力机制）
    ],
    lora_dropout=0 # LoRA dropout 概率，0 表示不使用 dropout
)


### 微调前的准备 ------------------------------------------------------------------
# 准备数据集
# 初始化 tokenizer
tokenizer = CLITokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer"
)

# 准备数据集
dataset = Text2ImageDataset(
    images_folder=images_folder,
    captions_folder=captions_folder,
    transform=train_transform,
    tokenizer=tokenizer,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    collate_fn=collate_fn,  # 之前定义的collate_fn()
    batch_size=train_batch_size,
    num_workers=8,
)
print("✅ 数据集准备完成！")

### 准备模型和优化器 --------------------------------------------------------------
'''
准备模型： 调用之前定义的 prepare_lora_model 函数。
准备优化器： 调用之前定义的 prepare_optimizer 函数。
设置学习率调度器： 使用 Hugging Face 的 get_scheduler 函数。
'''
# 准备模型
tokenizer, noise_scheduler, unet, vae, text_encoder = prepare_lora_model(
    lora_config,
    pretrained_model_name_or_path,
    model_path,
    resume=False,
    merge_lora=False
)

# 准备优化器
optimizer = prepare_optimizer(
    unet, 
    text_encoder, 
    unet_learning_rate=unet_learning_rate, 
    text_encoder_learning_rate=text_encoder_learning_rate
)

# 设置学习率调度器
lr_scheduler = get_scheduler(
    lr_scheduler_name,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
    num_training_steps=max_train_steps,
    num_cycles=num_cycles
)

print("✅ 模型和优化器准备完成！可以开始训练。")


### 开始微调 ---------------------------------------------------------------------
'''
- 训练循环： 我们在多个 epoch 中进行训练，直到达到 max_train_steps。每个 epoch 代表一轮数据的完整训练，在常见的 UI 界面中也可以看到 epoch 和 max_train_steps 的参数。
- 编码图像： 使用 VAE（变分自编码器）将图像编码为潜在表示（latent space），以便后续在扩散模型中添加噪声并进行处理。
- 添加噪声： 使用噪声调度器（noise_scheduler）为潜在表示添加随机噪声，模拟图像从清晰到噪声的退化过程。这是扩散模型的关键步骤，训练时模型通过学习如何还原噪声，从而在推理过程中通过逐步去噪生成清晰的图像。
- 获取文本嵌入： 使用文本编码器（text_encoder）将输入的文本 prompt 转换为隐藏状态（我们见过很多类似的表达：隐藏向量/特征向量/embedding/...），为图像生成提供文本引导信息。
- 计算目标值： 根据扩散模型的类型（epsilon 或 v_prediction），确定模型的目标输出（噪声或速度向量）。
- UNet 预测： 使用 UNet 模型对带噪声的潜在表示进行预测，生成的输出用于还原噪声或预测速度向量。
- 计算损失： 通过加权均方误差（MSE）计算模型损失，并进行反向传播。
- 优化与保存：通过优化器更新模型参数，并在适当时保存检查点。
'''

# 禁用并行化， 避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化 
global_step = 0
best_face_score = float("inf") # 初始化为正无穷大， 存储最佳面部相似度分数

# 进度条显示训练进度
progress_bar = tqdm(
    range(max_train_steps), # 根据 num_training_steps 设置
    desc="训练步骤"
)

# 训练循环
for epoch in range(math.ceil(max_train_steps / len(train_dataloader))):
    # 如果你想在训练中增加评估，那在循环中增加train() 是有必要的
    unet.train()
    text_encoder.train()

    for step, batch in enumerate(train_dataloader):
        if global_step >= max_train_steps:
            break

        # 编码图像为潜在表示（latent）
        latents = vae.encode(batch["pixel_values"].to(DEVICE, dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling.scaling_factor  # 根据 VAE 的缩放因子调整潜在空间

        # 为潜在表示添加噪声， 生成带噪声的图像
        noise = torch.ran_like(latents)  # 生成与潜在表示相同形状的随机噪声
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

         # 获取文本的嵌入表示
        encoder_hidden_states = text_encoder(batch["input_ids"].to(DEVICE))[0]

        # 计算目标值
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise  # 预测噪声
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)  # 预测速度向量

        # UNet 模型预测
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states)[0]

        # 计算损失
        if not snr_gamma:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # 计算信噪比 (SNR) 并根据 SNR 加权 MSE 损失
            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            
            # 计算加权的 MSE 损失
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        # 反向传播
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        global_step += 1

        # 打印训练损失
        if global_step % 100 == 0 or global_step == max_train_steps:
            print(f"🔥 步骤 {global_step}, 损失: {loss.item()}")

        # 保存中间检查点，当前简单设置为每 500 步保存一次
        if global_step % 500 == 0:
            save_path = os.path.join(output_folder, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)

            # 使用 save_pretrained 保存 PeftModel
            unet.save_pretrained(os.path.join(save_path, "unet"))
            text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
            print(f"💾 已保存中间模型到 {save_path}")

# 保存最终模型到 checkpoint-last
save_path = os.path.join(output_folder, "checkpoint-last")
os.makedirs(save_path, exist_ok=True)
unet.save_pretrained(os.path.join(save_path, "unet"))
text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
print(f"💾 已保存最终模型到 {save_path}")

print("🎉 微调完成！")