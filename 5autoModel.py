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

