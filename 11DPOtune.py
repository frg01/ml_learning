import os
import re
import json

import torch
import pandas as pd
from tqdm.auto import tqdm

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl import DPOConfig, DPOTrainer
"""
直接偏好优化（Direct Preference Optimization）：
目标：给定一个prompt，以及两个回答（一个“偏好”（chosen），一个“非偏好”（rejected）） 让模型学会生成偏好的回答
数学思维：优化概率偏好差距，其损失函数鼓励模型对偏好回答给出更高概率
三元组(prompt, chosen_response, rejected_response)
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
"""



# 数据集下载 git clone https://github.com/Baiiiiiiiiii/GenAI_hw6_dataset.git

with open("./GenAI_hw6_dataset/labelled_data.json", 'r') as jsonfile:
    full_data = json.load(jsonfile)

with open("./GenAI_hw6_dataset/test_prompt.json", 'r') as jsonfile:
    test_data = json.load(jsonfile)

model = AutoModelForCausalLM.from_pretrained(
    'MediaTek-Research/Breeze-7B-Instruct-v0_1',
    device_map='auto',
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('MediaTek-Research/Breeze-7B-Instruct-v0_1')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# 定义数据处理函数
def data_formulate(data):
    message = [
        {"role": "system", "content": '回覆請少於20字'},
        {"role": "user", "content": data['prompt']}
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return prompt

# 生成原始模型响应
original_model_response = []
for data in tqdm(test_data):
    id = data['id']
    print(f"Question {id}:\n{data['prompt']}")

    inputs = tokenizer(data_formulate(data), return_tensors="pt".to('cuda'))
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(**inputs,generation_config=generation_config)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
    original_model_response.append(output_text)

    print(f"Response from original model:\n{output_text}\n")


### 设置参数 ------------------------------------------------------------------------
num_epoch = 1        # 训练轮数
data_size = 50       # 用于训练的数据量
support_ratio = 0.1  # 偏好支持真人化的比例
'''
support_ratio 将反映人类的偏好：

0 表示完全不支持（反对）真人化
1 表示完全支持真人化
0.1 表示 10% 支持真人化， 90% 反对。
'''

# 准备训练数据 将数据集分为support 和 oppose 两部分，偏好对 （DPO）
# 选择部分数据用于训练
training_data = full_data[:data_size]

# 定义support 数据集的大小，用于一部份数据标记为“支持”
support_data_size = int(data_size * support_ratio)

# 为训练数据集准备数据
prompt_list = [data_formulate(data) for data in training_data]
chosen_list = [data['support'] for data in training_data[:support_data_size]] + [data['oppose'] for data in training_data[support_data_size:]]
rejected_list = [data['oppose'] for data in training_data[:support_data_size]] + [data['support'] for data in training_data[support_data_size:]] 
position_list = ['support' for _ in range(support_data_size)] + ['oppose' for _ in range(data_size - support_data_size)]

# 创建训练数据集
train_dataset = Dataset.from_dict({'prompt': prompt_list, 'position': position_list, 'chosen': chosen_list, 'rejected': rejected_list})
pd.DataFrame(train_dataset).rename(columns={"chosen": "preferred", "rejected": "non-preferred"})


### 训练 ------------------------------------------------------------------
# 设置训练参数
training_args = DPOConfig(
    output_dir='./',                     # 模型输出与检查点保存目录
    per_device_train_batch_size=1,      # 每个设备（GPU/CPU）上的训练批次大小
    num_train_epochs=num_epoch,         # 总训练轮数
    gradient_accumulation_steps=8,      # 梯度累积步数：相当于 batch_size * 8
    gradient_checkpointing=False,       # 是否开启梯度检查点，节省显存但牺牲一部分速度
    learning_rate=2e-4,                 # 初始学习率
    optim="paged_adamw_8bit",          # 使用 8-bit 分页 AdamW 优化器，节省显存
    logging_steps=1,                    # 每训练多少步记录一次日志
    warmup_ratio=0.1,                   # 预热比例：前 10% 训练步数线性增长学习率
    beta=0.1,                           # DPO 损失温度参数 β，控制偏好差异放大程度
    report_to='none',                   # 日志不上传到远程服务（如 WandB/TensorBoard）

    # 显式声明以避免警告
    max_length=512,                     # 最大生成长度（含 prompt）
    max_prompt_length=128,              # prompt 最大长度，超过截断
    remove_unused_columns=False,        # 保留数据集中所有列，避免自动裁剪
)


# 配置 PEFT (Parameter-Efficient Fine-Tuning)
peft_config = LoraConfig(
    lora_alpha=16,                      # LoRA 中 α 参数，控制权重缩放
    lora_dropout=0.1,                   # LoRA 层的 dropout 比例，以防过拟合
    r=64,                               # LoRA rank（矩阵分解秩），参数低秩近似维度
    bias="none",                      # 不对 bias 应用 LoRA
    task_type="CAUSAL_LM",            # 任务类型：因果语言模型微调
)

# 初始化DPO 训练器
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# 开始训练
dpo_trainer.train()

# 微调后的模型输出
trained_model_response = []
for data in tqdm(test_data): 
    id = data['id']
    print(f"Question {id}:\n{data['prompt']}")

    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id
    )
    output = model.generate(**inputs, generation_config=generation_config)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
    trained_model_response.append(output_text)

    print(f"Response from trained model:\n{output_text}\n")



### 微调前后模型响应
model_response = []
print(f"num_epoch: {num_epoch}\ndata_size: {data_size}\nsupport_ratio: {support_ratio}\n")

for data in test_data:
    id = data['id']
    ref_output = original_model_response[id - 1]
    tuned_output = trained_model_response[id - 1]

    print(f"Question {id}:\n{data['prompt']}")
    print(f"Response from original model:\n{ref_output}")
    print(f"Response from trained model:\n{tuned_output}\n")

    model_response.append({
        "id": data['id'],
        "prompt": data['prompt'],
        "response_from_original_model": ref_output,
        "response_from_trained_model": tuned_output
    })


