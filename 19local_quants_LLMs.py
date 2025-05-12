# LLM 的加载、微调、和应用及多个方面
# 首先聚焦加载， 正确安装和知晓模型量化的概念
"""
以下四个模型在huggingface都有量化版本（GPTQ/AWQ/GGUF）
mistralai/Mistral-7B-Instruct-v0.3
Qwen/Qwen2.5-7B-Instruct
meta-llama/Llama-3.1-8B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""
### 手动下载模型 --------------------------------------------------------
'''条件
sudo apt-get update
sudo apt-get install git git-lfs wget aria2
git lfs install

wget https://huggingface.co/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com

GPTQ:
./hfd.sh neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit --tool aria2c -x 16

AWQ：
./hfd.sh solidrust/Mistral-7B-Instruct-v0.3-AWQ --tool aria2c -x 16

GGUF：
./hfd.sh bartowski/Mistral-7B-Instruct-v0.3-GGUF --include Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --tool aria2c -x 16

库下载
pip install --upgrade transformers
pip install optimum
pip install numpy==1.26.4

pip list | grep auto-gptq
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定本地模型的路径
model_path = "./Mistral-7B-Instruct-v0.3-GPTQ-4bit"

# 指定远程模型的路径
model_path = "neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 下载并加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",  # 自动选择模型的权重数据类型
    device_map="auto"    # 自动选择可用的设备（CPU/GPU）
)

# 输入文本
input_text = "Hello, World!"

# 将输入文本编码为模型可接受的格式
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

# 生成输出
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=50,
    )

# 解码生成的输出
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 打印生成的文本
print(output_text)




### Prompt Template -----------------------------------------------------

# 定义 Prompt Template
prompt_template = "问：{question}\n答："

# 定义问题
question = "人工智能的未来发展方向是什么？"

# 使用 Prompt Template 生成完整的提示
prompt = prompt_template.format(question=question)
print(prompt)


