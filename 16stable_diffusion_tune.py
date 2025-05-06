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

