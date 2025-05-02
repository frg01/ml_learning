# bitsandbytes：用于低精度计算，加速训练过程。
# datasets：用于加载和处理数据集。
# transformers：Hugging Face 提供的库，包含预训练的模型和 Tokenizer。
# peft：Parameter-Efficient Fine-Tuning，参数高效微调库。
# sentencepiece：用于处理分词。
# accelerate：用于加速训练过程。
# colorama：用于在终端中打印彩色文本。
# fsspec：文件系统规范库。

import os,sys,argparse,json,warnings,logging
import torch
import torch.nn as nn
import bitsandbytes as bnb 
from datasets import load_dataset,load_from_disk
import transformers 
from peft import PeftModel
from colorama import Fore, Style

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitAndBytesConfig,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)


model_name = "MediaTek-Research/Breeze-7B-Instruct-v0_1"


### 数据预处理-------------------------------------------------------------------------------------------------
def generate_training_data(data_point):
    '''
    将输入和输出文本转换为模型可读取的tokens。
    参数:
    - data_point: 包含 “instruction"、“inputs”、“output”字段的字典
    返回：
    - 包含模型输入IDs、标签和注意力掩码的字典。
    示例：
    - 如果你构建一个字典 data_point_1 ,并包含字段 “instrcution”、“inputs”、“output”，你可以像这样使用函数
        generate_training_data(data_point_1)
    '''
    # 构建完整的输入提示词
    prompt = f"""\
    [INST] <<SYS>>
    You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
    <</SYS>>

    {data_point["instruction"]}
    {data_point["input"]}
    [/INST]"""

    #计算用户提示词的toekn数量
    len_user_prompt_tokens = (
        len(
            tokenizer(
                prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        ) - 1
    )

    #将完整的输入和输出转换为tokens
    full_toekns = tokenizer(
        prompt + " " + data_point["output"] + "</s>",
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]

    return {
        "input_ids":full_toekns,
        "labels": [-100] * len_user_prompt_tokens + full_toekns[len_user_prompt_tokens:],
        "attention_mask": [1] * len(full_toekns),
    }


### 模型评估-------------------------------------------------------------------------------------------------
def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    """
    获取模型在给定输入下的生成结果。
    参数：
    - instruction： 描述任务的字符串。
    - generation_config: 模型生成配置。
    - max_len: 最大生成长度。
    - verbose： 是否打印生成结果。

    返回：
    - output： 模型生成的文本。
    """

    # 构建完整的输入提示词
    prompt = f"""\
    [INST] <<SYS>>
    You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
    <</SYS>>

    {instruction}
    {input_text}
    [/INST]"""

    # 将提示词转换为模型所需的token格式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # 使用模型生成回复
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )

    # 解码并打印生成的回复
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split["[/INST]"][1].replace("</s>", "").replace("<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
        if verbose:
            print(output)
    
    return output



### 选择预训练模型-------------------------------------------------------------------------------------------------
model_name = "MediaTek-Research/Breeze-7B-Instruct-v0_1"

cache_dir = "./cache"

nf4_config = BitAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 从指定模型名称或路径加载预训练语言模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)

#创建tokenizer 并设置结束符号 （eos_token）
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token

#设置模型推理时的解码参数
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,                # 启用采样（即非贪婪或束搜索），会根据概率随机生成 token
    temperature=0.1,               # 温度系数，控制采样的“随机性”，越低越确定
    num_beams=1,                   # 束搜索的数量，=1 表示关闭 beam search，仅依赖采样
    top_p=0.3,                     # nucleus sampling，保留概率累计为 top_p 的 token 子集（更强筛选）
    no_repeat_ngram_size=3,       # 禁止重复的 3-gram，降低重复率
    pad_token_id=2,               # 设置用于 padding 的 token ID，常与 tokenizer 配套使用
)



### 初始表现-------------------------------------------------------------------------------------------------
""" 样例和 Prompt 都保持繁体 """

# 测试样例
test_tang_list = [
    '相見時難別亦難，東風無力百花殘。',
    '重帷深下莫愁堂，臥後清宵細細長。',
    '芳辰追逸趣，禁苑信多奇。'
]

# 获取每个样例的模型输出
demo_before_finetune = []

for tang in test_tang_list:
    demo_before_finetune.append(
        f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n' + 
        evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose=False)
    )

for idx in range(len(demo_before_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_before_finetune[idx])
    print("-" * 80)


### 设置用于微调的超参数-------------------------------------------------------------------------------------------------
""" 强烈建议你尝试调整这个参数 """

num_train_data = 1040  # 设置用于训练的数据量，最大值为5000。通常，训练数据越多越好，模型会见到更多样化的诗句，从而提高生成质量，但也会增加训练时间。
                      # 使用默认参数(1040)：微调大约需要25分钟，完整运行所有单元大约需要50分钟。
                      # 使用最大值(5000)：微调大约需要100分钟，完整运行所有单元大约需要120分钟。
        
""" 你可以（但不一定需要）更改这些超参数 """

output_dir = "./output"  # 设置作业结果输出目录。
ckpt_dir = "./exp1"  # 设置 model checkpoint 保存目录（如果想将 model checkpoints 保存到其他目录下，可以修改这里）。
num_epoch = 1  # 设置训练的总 Epoch 数（数值越高，训练时间越长，若使用免费版的 Colab 需要注意时间太长可能会断线，本地运行不需要担心）。
LEARNING_RATE = 3e-4  # 设置学习率

""" 建议不要更改此单元格中的代码 """

cache_dir = "./cache"  # 设置缓存目录路径
from_ckpt = False  # 是否从 checkpoint 加载模型权重，默认值为否
ckpt_name = None  # 加载特定 checkpoint 时使用的文件名，默认值为无
dataset_dir = "./GenAI-Hw5/Tang_training_data.json"  # 设置数据集目录或文件路径
logging_steps = 20  # 定义训练过程中每隔多少步骤输出一次日志
save_steps = 65  # 定义训练过程中每隔多少步骤保存一次模型
save_total_limit = 3  # 控制最多保留多少个模型 checkpoint
report_to = "none"  # 设置不上报实验指标，也可以设置为 "wandb"，此时需要获取对应的 API，见：https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/pull/5
MICRO_BATCH_SIZE = 4  # 定义微批次大小
BATCH_SIZE = 16  # 定义一个批次的大小
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 计算每个微批次累积的梯度步骤
CUTOFF_LEN = 256  # 设置文本截断的最大长度
LORA_R = 8  # 设置 LORA（Layer-wise Random Attention）的 R 值
LORA_ALPHA = 16  # 设置 LORA 的 Alpha 值
LORA_DROPOUT = 0.05  # 设置 LORA 的 Dropout 率
VAL_SET_SIZE = 0  # 设置验证集的大小，默认值为无
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]  # 设置目标模块，这些模块的权重将被保存为 checkpoint。
device_map = "auto"  # 设置设备映射，默认值为 "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获取环境变量 "WORLD_SIZE" 的值，若未设置则默认为 1
ddp = world_size != 1  # 根据 world_size 判断是否使用分布式数据处理(DDP)，若 world_size 为 1 则不使用 DDP
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


### 开始微调-------------------------------------------------------------------------------------------------
# 设置TOKENIZERS_PARALLELISM为false，这里简单禁用并行性以避免报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 创建指定的输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# 根据 from_ckpt 标志，从 checkpoint 加载模型权重
if from_ckpt:
    model = PeftModel.from_pretrained(model, ckpt_name)

# 对量化模型进行预处理以进行训练
model = prepare_model_for_kbit_training(model)

# 使用 LoraConfig 配置 LORA 模型
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# 将tokenizer 的填充token设置为0
tokenizer.pad_token_id = 0

# 加载并处理训练数据
with open(dataset_dir, "r", encoding="utf-8") as f:
    data_json = json.load(f)

with open("tmp_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data_json[:num_train_data], f, indent=2, ensure_ascii=False)

data = load_dataset('json', data_files="tmp_dataset.json", download_mode="force_redownload")

# 将训练数据分为训练集和验证集（若 VAL_SET_SIZE 大于 0）
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )

    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else: 
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None

# 使用Transformers Trainer 进行模型训练
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=True,  # 使用混合精度训练
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,  # 是否使用 DDP，控制梯度更新策略
        report_to=report_to,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 禁用模型的缓存功能
model.config.use_cache = False

# 若使用 PyTorch 2.0 以上版本且非 Windows 系统，编译模型
if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

# 开始模型训练
trainer.train()

# 将训练好的模型保存到指定目录
model.save_pretrained(ckpt_dir)

# 打印训练过程中可能出现的缺失权重警告信息
print("\n 如果上方有关于缺少键的警告，请忽略 :)")


### 测试微调后的模型------------------------------------------------------------------------
# 查看所有可用的checkpoints
ckpts =[]
for ckpt in os.listdir(ckpt_dir):
    if ckpt.startswith("checkpoint-"):
        ckpts.append(ckpt)

# 列出所有的checkpoints
ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[-1]))
print("所有可用的 checkpoints：")
print(" id: checkpoint 名称")
for (i, ckpt) in enumerate(ckpt):
    print(f"i:>3: {ckpt}")

""" 你可以（但不一定需要）更改 checkpoint """

id_of_ckpt_to_use = -1  # 要用于推理的 checkpoint 的 id（对应上一单元格的输出结果）。
                        # 默认值 -1 表示使用列出的最后一个 checkpoint。
                        # 如果你想选择其他 checkpoint，可以将 -1 更改为列出的 checkpoint id 中的某一个。

ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])

""" 你可以（但不一定需要）更改解码参数 """
# 你可以在此处调整解码参数，解码参数的详细解释请见作业幻灯片。
max_len = 128  # 生成回复的最大长度
temperature = 0.1  # 设置生成回复的随机度，值越小生成的回复越稳定。
top_p = 0.3  # Top-p (nucleus) 采样的概率阈值，用于控制生成回复的多样性。
# top_k = 5  # 调整 Top-k 值，以增加生成回复的多样性并避免生成重复的词汇。

# 释放显存
import gc
# 删除模型和 tokenizer 对象
del model
del tokenizer

# 调用垃圾回收机制，强制释放未使用的内存
gc.collect()

# 清理 GPU 缓存
torch.cuda.empty_cache()

### 加载模型和分词器 ---------------------------------------------------------------------------
test_data_path = "GenAI-Hw5/Tang_testing_data.json"  # 测试数据集的路径
output_path = os.path.join(output_dir, "results.txt")  # 生成结果的输出路径

cache_dir = "./cache"
seed = 42 
no_repeat_ngram_size = 3

# 配置模型的量化设置
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},
    cache_dir=cache_dir
)

# 加载微调后的权重
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

### 生成测试结果 ----------------------------------------------------------------------
results = []

# 设置生成配置，包括随机度、束搜索等参数
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,  # 如果需要使用 top-k，可以在此设置
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=2
)

# 读取测试数据集
with open(test_data_path,"r",encoding="utf-8") as f:
    test_datas = json.load(f)

# 对每个测试样例生成预测，并保存结果
with open(output_path,"w",encoding="utf-8") as f:
    for (i, test_data) in enumerate(test_datas):
        predict = evaluate(test_data["instruction"], generation_config, max_len,test_data["input"],verbose=False)
        f.write(f"{i+1}. " + test_data["input"] + predict + "\n")
        print(f"{i+1}. " + test_data["input"] + predict)


# 使用之前的测试例子
test_tang_list = [
    '相見時難別亦難，東風無力百花殘。',
    '重帷深下莫愁堂，臥後清宵細細長。',
    '芳辰追逸趣，禁苑信多奇。'
]

# 使用微调后的模型进行推理
demo_after_finetune = []
for tang in test_tang_list:
    demo_after_finetune.append(
        f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n' +
        evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose=False)
    )

# 打印输出结果
for idx in range(len(demo_after_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_after_finetune[idx])
    print("-" * 80)