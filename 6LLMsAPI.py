# 模型部署为一个API服务，供其他应用调用
# 两个选择：Flask和FastAPI(Flask简短易用，性能相对低，异步支持弱.FastAPI性能高，异步支持好)
# uvicorn 6LLMsAPI:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#定义请求体的数据模型
class PromptRequest(BaseModel):
    prompt: str
    # max_length: int = 50
    # num_beams: int = 5
    # no_repeat_ngram_size: int = 2
    # early_stopping: bool = True
    # temperature: float = 0.7
    # top_p: float = 0.95
    # top_k: int = 50 

app = FastAPI()

#加载模型和分词器
model_name = "distilgpt2" #文本补全
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
model.to(device)

@app.post("/generate")
def generate_text(request: PromptRequest):
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=200,  # 最大生成长度
            num_beams=5,    # Beam Search的数量，使用束搜索，保留前5个可能性继续探索，选出一个最优的
            no_repeat_ngram_size=2,# 防止重复n-gram，防止重复话语
            early_stopping=True # 提前停止，遇到<eos>标记就停止，防止继续废话
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

### 发送请求
# import requests

# response = requests.post(
#     "http://localhost:8000/generate",
#     json={"prompt": "Hello GPT"}
# )
# print(response.json())





# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# app = Flask(__name__)

# # 加载模型和分词器
# model_name = "distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# @app.route('/generate', methods=['POST'])
# def generate():
#     prompt = request.json.get('prompt')
#     if not prompt:
#         return jsonify({'error': 'No prompt provided'}), 400

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_length=200,
#             num_beams=5,
#             no_repeat_ngram_size=2,
#             early_stopping=True
#         )
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({'generated_text': generated_text})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)