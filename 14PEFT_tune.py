# PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的专门用于参数高效微调的工具库。

'''微调方法
Prefix-Tuning：冻结原模型参数，为每一层添加可学习的前缀向量，只学习前缀参数。
Adapter-Tuning：冻结原模型参数，在模型的层与层之间插入小型的 adapter 模块，仅对 adapter 模块进行训练。
等。。。
'''

### 加载预训练模型 --------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练的 GPT-2 的分词器
tokenizer = AutoTokenizer.from_pretrained('opus-mt-zh-en')
model = AutoModelForCausalLM.from_pretrained('opus-mt-zh-en')

# print(model)


### 应用LoRA --------------------------------------------------------------------
from peft import get_peft_model, LoraConfig, TaskType

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型：因果语言模型
    inference_mode=False,         # 推理模式关闭，以进行训练
    r=8,                          # 低秩值r
    lora_alpha=32,                # LoRA 的缩放因子
    lora_dropout=0.1,             # Dropout 概率
    target_modules=["q_proj", "v_proj"],  # 💡 关键参数
)

# 将 LoRA 应用到模型中
model = get_peft_model(model, lora_config)
# print(model)

# 查看 LoRA 模块
model.print_trainable_parameters()


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"可训练参数量: {trainable_params}")
    print(f"总参数量: {all_params}")
    print(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")
    
# print_trainable_parameters(model)


### 准备数据进行微调 --------------------------------------------------------------------
from transformers import Trainer, TrainingArguments
# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',         # 模型保存和日志输出的目录路径
    num_train_epochs=3,             # 训练的总轮数（epochs）
    per_device_train_batch_size=16, # 每个设备（如GPU或CPU）上的训练批次大小，16表示每次输入模型的数据数量
    learning_rate=5e-5,             # 学习率
    logging_steps=10,               # 每隔多少步（steps）进行一次日志记录
    save_steps=100,                 # 每隔多少步保存模型
)

from datasets import Dataset
# 假设我们想训练中翻英
raw_data = [
    {"translation": {"zh": "你好", "en": "Hello"}},
    {"translation": {"zh": "今天天气很好", "en": "The weather is nice today"}},
]

# 编码为模型训练输入格式（因 CausalLM 只看 input_ids）
def preprocess(example):
    input_text = example["translation"]["zh"]
    target_text = example["translation"]["en"]
    combined = input_text + " </s> " + target_text  # 你也可以只输入 source
    tokenized = tokenizer(combined, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = Dataset.from_list(raw_data).map(preprocess)

# 创建 Trainer
trainer = Trainer(
    model=model,                    # 训练的模型对象，需要事先加载好
    args=training_args,             # 上面定义的训练参数配置
    train_dataset=train_dataset,    # 需要对应替换成已经处理过的dataset
)

# 开始训练
trainer.train()

### 保存和加载LoRA 微调的模型 ----------------------------------------------------------
#保存LoRA参数
model.save_pretrained('./lora_model')

# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained("opus-mt-zh-en")

# 加载LoRA 参数
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, './lora_model')


### 合并LoRA 并卸载PEFT包装（减少依赖成为标准模型、提高推理效率、简化模型保存和加载）------------
# 对比合并前后的模型
print("合并前的模型结构：")
print(model)

# 合并，并卸载LoRA权重
model = model.merge_and_unload()

print("合并后的模型结构")
print(model)


# 保存合并后的模型
model.save_pretrained('./merged_model')
tokenizer.save_pretrained('./merged_model')