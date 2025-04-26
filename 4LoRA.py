# 从线性层搭配注意力机制 LoRA - low-rank adaptation,微调大型预训练模型的技术
# 核心思想：通过低秩分解减少微调时的参数量，而不牺牲模型的性能，文章地址 https://arxiv.org/abs/2106.09685
#冻结原模型参数，只训练插入的低秩矩阵
#公式： y=(W+ΔW)x + b, ΔW=BA，
#A: shape 是r * k  (r是低秩矩阵的秩，k是原矩阵的列数)
#B: shape 是d * r  (d是原矩阵的行数)  



## 线性层的LoRA
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super(LoRALinear, self).__init__()
        self.in_features = in_features #对应d
        self.out_features = out_features #对应k
        self.r = r  #低秩值

        # 原始权重矩阵，冻结
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False # 冻结

        # LoRA部分的参数，初始化A 从均值0的正太分布中采样，B为全零
        self.A = nn.Parameter(torch.empty(r, in_features)) #形状（r,d）
        self.B = nn.Parameter(torch.zeros(out_features, r)) #形状 （k,r）
        nn.init.normal_(self.A, mean=0.0, std=0.01) #初始化A

        # LoRA的偏置项，可选
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 原始部分
        original_output = torch.nn.functional.linear(x, self.weight, self)
        # LoRA 增量部分
        delta_W = torch.matmul(self.B, self.A) # 形状为 （k,d）
        lora_output = torch.nn.functional.linear(x, delta_W)
        
        #总输出
        return original_output + lora_output

# 单头注意力LoRA
class LoRAAttention(nn.Module):
    def __init__(self, embed_dim, r=4):
        super(LoRAAttention, self).__init__()
        self.embed_dim = embed_dim
        self.r = r

        # 原始QKV权重矩阵，冻结
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)

        # 冻结
        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            param.requires_grad = False

        # LoRA 的 Q部分
        self.A_Q = nn.Parameter(torch.empty(r, embed_dim)) #形状（r,d）
        self.B_Q = nn.Parameter(torch.zeros(embed_dim, r)) #形状 （k,r）
        nn.init.normal_(self.A_Q, mean=0.0, std=0.02)
        # LoRA 的 K部分
        self.A_K = nn.Parameter(torch.empty(r, embed_dim)) #形状（r,d）
        self.B_K = nn.Parameter(torch.zeros(embed_dim, r)) #形状 （k,r）
        nn.init.normal_(self.A_K, mean=0.0, std=0.02)
        # LoRA 的 V部分
        self.A_V = nn.Parameter(torch.empty(r, embed_dim)) #形状（r,d）
        self.B_V = nn.Parameter(torch.zeros(embed_dim, r)) #形状 （k,r）
        nn.init.normal_(self.A_V, mean=0.0, std=0.02)


    def forward(self, query, key, value):
        """
        query, key, value: 形状为*（batch_size,seq_length, embed_dim）
        """
        #计算原始的Q、K、V
        Q = self.W_Q(query)# (batch_size, seq_length, embed_dim)
        K = self.W_K(key)
        V = self.W_V(value)

        #计算LoRA的增量部分
        delta_Q = torch.matmul(query, self.A_Q.t()) # (batch_size, seq_length, r)
        delta_Q = torch.matmul(delta_Q, self.B_Q.t()) # (batch_size, seq_length, embed_dim)
        delta_K = torch.matmul(key, self.A_K.t()) # (batch_size, seq_length, r)
        delta_K = torch.matmul(delta_K, self.B_K.t())
        delta_V = torch.matmul(value, self.A_V.t())
        delta_V = torch.matmul(delta_V, self.B_V.t())

        #更新后的Q、K、V
        Q = Q + delta_Q
        K = K + delta_K
        V = V + delta_V

        #计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        #输出层
        output = self.W_O(context)

        return output