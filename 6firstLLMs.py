# 选择一个小模型 distilgpt2

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型名称
model_name = "distilgpt2"

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)

#模型移动到GPU或mps
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
model.to(device)

# 进行推理
# 设置模型为评估模式
model.eval()

# 输入文本
input_text = "Hello. How are you?"

# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()} 

# 生成文本
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_length=50,  # 最大生成长度
        num_beams=5,    # Beam Search的数量，使用束搜索，保留前5个可能性继续探索，选出一个最优的
        no_repeat_ngram_size=2,# 防止重复n-gram，防止重复话语
        early_stopping=True # 提前停止，遇到<eos>标记就停止，防止继续废话
    )

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  
print("模型生成的文本：")
print(generated_text)

