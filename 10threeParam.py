# Top-K、Top-P、Temperature 

'''
Greedy Search（贪心搜索）: 每次选择概率最高的词汇。
Beam Search（束搜索）: 保留多个候选序列，平衡生成质量和多样性。
Top-K 采样: 限制候选词汇数量。
Top-P 采样（Nucleus Sampling）: 根据累积概率选择候选词汇，动态调整词汇集。
'''

"""
Top-K 采样： 选择概率最高的 K 个词汇，重新归一化，使其总和为1。随机采样一个词汇作下一个生成词。
Top-P 采样： 选择累积概率达到P的集合，重新归一化，随机采样一个作为下一个生成的词。
Temperature：控制生成文本随机性，改变生成概率的分布锐度。值越大，回答内容越随机。
"""

### Top-K 采样----------------------------------------------------
import numpy as np

# 概率分布
probs = np.array([0.4,0.3,0.2,0.05,0.05])
words = ['A','B','C','D','<eos>']

#设置Top-K
K=3

# 获取概率最高的 K 个词汇索引
top_indices = np.argsort(probs)[-K:]

# 保留这些 K 个词汇及其概率
top_k_probs = np.zeros_like(probs)
top_k_probs[top_indices] = probs[top_indices]


# 归一化保留的 K 个词汇的概率
top_k_probs = top_k_probs / np.sum(top_k_probs)  #[0.4,0.3,0.2,0.,0.] / 0.9


# # 打印 Top-k 采样的结果
# print("Top-K 采样选择的词汇和对应的概率：")
# for i in top_indices:
#     print(f"{words[i]}: {top_k_probs[i]:.2f}")



### Top-P 采样 ----------------------------------------------------------------------------
# 概率分布
probs = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
words = ['A', 'B', 'C', 'D', '<eos>']

# 设置Top-p
P=0.6

# 对概率进行排序
sorted_indices = np.argsort(probs)[::-1] # 从大到小排序
sorted_probs = probs[sorted_indices]

# 累计概率
cumulative_probs = np.cumsum(sorted_probs)

# 找到累计概率大于等于P的索引
cutoff_index = np.where(cumulative_probs >= P)[0][0]

# 保留累积概率达到 P 的词汇及其概率
top_p_probs = np.zeros_like(probs)
top_p_probs[sorted_indices[:cutoff_index + 1]] = sorted_probs[:cutoff_index + 1]

# 归一化保留的词汇的概率
top_p_probs = top_p_probs / np.sum(top_p_probs)

# # 打印 Top-P 采样的结果
# print("\nTop-P 采样选择的词汇和对应的概率：")
# for i in np.where(top_p_probs > 0)[0]:
#     print(f"{words[i]}: {top_p_probs[i]:.2f}")



### Temperature ----------------------------------------------------------
import matplotlib.pyplot as plt

# 概率分布
probs = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
words = ['A', 'B', 'C', 'D', '<eos>']

# 设置 Top-K
K = 5

# 设置不同的 Temperature 值
temperatures = [0.5, 1.0, 1.5]

plt.figure(figsize=(10,6))

# 遍历不同的温度
for temp in temperatures:
    # 使用Temperature 调整概率
    """
    a ** b , 
    0 <= b < 1：b越大越接近原概率，b越小结果结果越接近1
    b >= 1: b越大，结果越小
    """
    adjusted_probs = probs ** (1.0 / temp)
    print(f"adjust:{adjusted_probs}")
    adjusted_probs = adjusted_probs / np.sum(adjusted_probs) # 归一化

    # 打印当前Temperature 的概率分布
    print(f"\n--- Temperature = {temp} ---")
    for i, prob in enumerate(adjusted_probs):
        print(f"{words[i]}: {prob:.2f}")
    
    # 绘制概率分布图
    plt.plot(words, adjusted_probs, label=f"Temperature = {temp}")

# 绘制原始概率分布的对比
plt.plot(words, probs, label="Original", linestyle="--", color="black")

# 添加图表信息
plt.xlabel("Word")
plt.ylabel("Probability")
plt.title("Effect of Temperature on Top-K Probability Distribution")
plt.legend()

# 显示图表
plt.show()