# Inseq 特征归因 understand what generative AI is thinking 
"""条件
pip install inseq
pip install transformers
pip install bitsandbytes
pip install accelerate
pip install sacremoses

wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh 'Helsinki-NLP/opus-mt-zh-en' --tool aria2c -x 16
"""

# 使用 Inseq 来了解输入文本的哪些部分对模型生成下一个单词的影响最大。
import inseq 
# 定义要使用的归因方法列表(梯度归因，注意权重) 调试、解释、偏差检测
attribution_methods = ['saliency', 'attention']

for method in attribution_methods:
    print(f"=======归因方法：{method}=======")
    # 加载中译英模型并设置归因方法
    # model = inseq.load_model("Helsinki-NLP/opus-mt-zh-en", method)
    model = inseq.load_model("opus-mt-zh-en", method)  # 导入之前下载到本地的模型

    # 使用指定的归因方法对输入文本进行归因
    attribution_result = model.attribute(
        input_texts="我喜欢机器学习和人工智慧。",
    )

    # 从tokenizer种去除 '——'前缀以避免混淆
    for attr in attribution_result.sequence_attributions:
        for item in attr.source:
            item.token = item.token.replace('_', '')
        for item in attr.target:
            item.token = item.token.replace('_', '')

    # 显示归因结果
    attribution_result.show()
