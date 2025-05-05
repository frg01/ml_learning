### 深入理解Beam Search（集束搜索/束搜索）------------------------------------------------------------
# 可以当作贪心算法的扩展，贪心算法只选最好的，而Beam Search会在多个候选中进行选择
# Greedy Search 每一步都选择当前概率最大的词
"""
 Beam Width（束宽）：num_beams
 所有候选序列生成结束标记 ：early_stopping
 最大生成长度: max_length
 长度惩罚: length_penalty
 防止生成序列中出现重复的 n-gram: no_repeat_ngram_size
 指定生成的序列数量: num_return_sequences
"""

#工作原理： 保留K个候选序列，计算累计概率或对数概率
#基本步骤：
# 1. 初始化：输出vocab_size长度的概率分布，选择最高的K个
# 2. 迭代生成、选择顶束： 使用k个词，得到k * vocab_size个链路，选出（log 概率）累积值最高的 top-k 条，不断迭代重复。
# 3. 终止条件： 当所有候选序列生成终止标记，(如 <eos>）或达到最大长度T，停止生成
# 4. 选择最终序列：选择得分最高的序列作为输出。
'''
Step 0:         <BOS>
                ↓
Step 1:       A       B          ← 只保留 2 条
              ↓       ↓
Step 2:   A1 A2 A3   B1 B2 B3     ← 每个扩展 vocab_size 个（这里只画 3）
              ↓       ↓
Step 3:   只保留概率最高的2条继续……
'''

import math 

def beam_search(initial_sequence, beam_width, max_length, vocab, get_next_probs):
    beam = [(initial_sequence, 0.0)] # (sequence, log_prob)
    completed = []

    for step in range(max_length):
        print(f"\n第 {step + 1} 步:")
        all_candidates = []
        for seq, score in beam:
            if seq.endswith('<eos>'):
                completed.append((seq, score))
                print(f"已完成序列: {seq}，得分为 {score}")
                continue
            next_probs = get_next_probs(seq)
            print(f"扩展序列: {seq}，当前得分为 {score}")
            for token, prob in next_probs.items():
                new_seq = seq + token 
                new_score = score + math.log(prob)
                all_candidates.append((new_seq,new_score))
                print(f"  候选序列: {new_seq}，得分为 {new_score}")
        # 对所有候选序列按得分降序排列，选择得分最高的 beam_width 个序列
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beam = all_candidates[:beam_width]

        # 打印选出的顶束序列
        print(f"\n选择的 {beam_width} 个顶束序列:")
        for seq, score in beam:
            print(f"  {seq}，得分为 {score}")

        # 如果没有更多序列可以扩展，则退出循环
        if not beam:
            break

    # 将当前beam中剩下的序列加入完成序列中
    completed += beam

    # 对完成的序列按得分降序排列，选择得分最高的序列
    completed.sort(key=lambda x: x[1],reverse=True)

    print("\n已完成的所有序列:")
    for seq, score in completed:
        print(f"  {seq}，得分为 {score}")
    
    return completed[0][0]



# 我们之前示例中设置的概率
def get_next_probs(seq):
    probs = {
        "": {"A": 0.4, "B": 0.3, "C": 0.2, "<eos>": 0.1},
        "A": {"A": 0.3, "B": 0.1, "C": 0.4, "<eos>": 0.2},
        "B": {"A": 0.1, "B": 0.1, "C": 0.3, "<eos>": 0.5},
        "AC": {"A": 0.1, "B": 0.2, "C": 0.5, "<eos>": 0.2},
    }
    return probs.get(seq, {"<eos>": 1.0})

initial_sequence = ""
beam_width = 2    #可以修改这个参数来感受区别
max_length = 5 
vocab = {"A", "B", "C", "<eos>"}

# best_sequence =  beam_search(initial_sequence, beam_width, max_length, vocab, get_next_probs)
# print("\n最佳序列:", best_sequence)


### 实际应用 ------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

# 指定模型名称
model_name = "distilgpt2"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 移动模型到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置模型为评估模式
model.eval()

# 输入文本
input_text = "Hello GPT"

# 编码输入文本 ，同时返回attention_mask
inputs = tokenizer.encode_plus(input_text, return_tensors="pt",padding=True).to(device)

# 生成文本， 使用Beam Search
beam_width = 5

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        num_beams=beam_width, # 你可以看到 beam_width 对应的参数名为 num_beams
        no_repeat_ngram_size=2,
        early_stopping=True, #当所有候选序列生成<eos>停止
        pad_token_id=tokenizer.eos_token_id
    )

# 解码生成的文本
generate_text = tokenizer.decode(outputs[0], skip_special_token=True)
print("生成的文本：")
print(generate_text)


### 对比不同的束宽 -----------------------------------------------------------------------

# 设置束宽不同的生成策略
beam_widths = [1, 3, 5]  
# 生成并打印结果
for beam_width in beam_widths:
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            num_beams=beam_width,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"束宽 {beam_width} 的生成结果：")
    print(generated_text)
    print('-' * 50)