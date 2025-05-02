#选择合适的AutoModel类
# 文本生成：使用 AutoModelForCausalLM。
# 填空任务：使用 AutoModelForMaskedLM。
# 机器翻译、文本摘要：使用 AutoModelForSeq2SeqLM。
# 抽取式问答：使用 AutoModelForQuestionAnswering（详见：《22a. 微调 LLM：实现抽取式问答》）
# 命名实体识别：使用 AutoModelForTokenClassification。
# 文本分类：使用 AutoModelForSequenceClassification。
# 特征提取或自定义任务：使用 AutoModel





"""
# GPT-2模型的配置文件
{
  "activation_function": "gelu_new", # 激活函数
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1, #注意力模块
  "bos_token_id": 50256, # 起始标记ID
  "embd_pdrop": 0.1, #嵌入模块
  "eos_token_id": 50256,
  "initializer_range": 0.02,# 权重初始化范围
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2", 
  "n_ctx": 1024, #最大上下文长度
  "n_embd": 768,# 嵌入向量大小
  "n_head": 12, # 注意力头数
  "n_layer": 12,# 堆叠的注意力模块层数
  "n_positions": 1024, # 位置编码范围
  "resid_pdrop": 0.1,#残差模块
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": { #
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },#生成任务参数（启用随机采样，最大生成长度）
  "vocab_size": 50257 # 词汇表大小
}
"""

### 示例1 文本生成
from transformers import AutoTokenizer, AutoModelForCausalLM

#指定模型名称
model_name = "gpt2"

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载语预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入文本
input_text = "Hello, how are you?"
#编码输入文本
imputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(**imputs, max_length=50, do_sample=True, top_p=0.95,temperature=0.7)

#解码生成的文本
generate_text = tokenizer.decode(outputs[0],skip_special_tokens=True)
print(generate_text)



###示例2 填空任务(AutoModelForMaskedLM)
import torch 
from transformers import AutoTokenizer, AutoModelForMaskedLM

#指定模型名称
model_name = "bert-base-uncased"

#加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#加载预训练模型
model = AutoModelForMaskedLM.from_pretrained(model_name)
#输出文本，包含【Mask】标记
input_text = "The capital of France is [MASK]."

#编码输入
inputs = tokenizer(input_text,return_tensors="pt")

# 获取预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 获取最高得分的漏洞
masked_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1).item()
predicted_token = tokenizer.decode([predicted_token_id])

print(f"预测结果: {predicted_token}")


### 示例3 序列到序列任务(AutoModelForSeq2SeqLM)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#指定模型名称
model_name = "Helsinki-NLP/opus-mt-en-fr"
#加载Tokenizer
toeknizer = AutoTokenizer.from_pretrained(model_name)

#加载预训练模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#输入文本
input_text = "Hello, how are you?"
# 编码
inputs = tokenizer(input_text, return_tensors="pt")

# 生成翻译
outputs = model.generate(**inputs, max_length=40, num_beams=4, early_stopping=True)

# 解码生成的文本
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"翻译结果: {translated_text}")


# 问答系统（AutoModelForQuestionAnswering）
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#指定模型名称
model_name = "distilbert-base-uncased-distilled-squad"

#加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#加载预训练模型
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#输入上下文和问题
context = "Hugging Face is creating a tool that democratizes AI."
question = "What is Hugging Face creating?"

#编码输入
inputs = tokenizer(question, context, return_tensors="pt")

#获取预测 no_grad()表示不计算梯度
with torch.no_grad():
    outputs = model(**inputs)

# 获取答案的起始和结束位置
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

# 解码答案
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
print(f"答案: {answer}")

### 示例5 命名实体识别（AutoModelForTokenClassification）从一句话中找出人名，组织名，地点名等
# flowchart TD
#     A(输入一段文本) --> B(Tokenizer编码成张量)
#     B --> C(模型前向推理得到logits)
#     C --> D(对每个token取最高分的类别)
#     D --> E(把类别id映射成标签名字)
#     E --> F(打印出token和对应标签)

from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

#指定模型名称
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

#加载Tokenizer 分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

#加载预训练模型
model = AutoModelForTokenClassification.from_pretrained(model_name)

#标签列表
label_list = model.config.id2label 

#输入文本
input_text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."

#编码输入 inputs 中包含input_ids, attention_mask等
inputs = tokenizer(input_text, resturn_tensors="pt") # pt返回张量格式

#获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

#获取预测分数
logits = outputs.logits
predictions = np.argmax(logits, dim=2)

# 将预测结果映射到标签
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
pred_labels = [label_list[prediction.item()] for prediction in predictions[0]]

#打印结果
for token, label in zip(tokens, pred_labels):
    print(f"{token}: {label}")

# 文本分类 （AutoModelForSequenceClassfication）
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

#指定模型名称
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#输入文本
input_text = "I love using transformers library!"

#编码输入
inputs = tokenizer(input_text, return_tensors="pt")

#获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测分数
logits = outputs.logits
probabilites = F.softmax(logits, dim=1)

# 获取预测标签
labels = ['Negative', 'Positive']
prediction = torch.argmax(probabilites, dim=1)
predicted_label = labels[prediction]

# 打印结果
print(f"文本：{input_text}")
print(f"预测标签：{predicted_label}")


### 示例7 特征提取
from transformers import AutoTokenizer, AutoModel

# 指定模型名称
model_name = "bert-base-uncased"

# 加载Tokenizer和模型
toeknizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 输入文本
input_text = "This is a sample sentence."

# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 获取最后一层隐藏状态
last_hidden_states = outputs.last_hidden_state

# 输出维度
print(f"最后一层隐藏状态的维度: {last_hidden_states.shape}")

