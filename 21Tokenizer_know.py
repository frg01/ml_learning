# 什么是tokenizer？在模型输入的编码和解码阶段发挥着重要作用
"""
模型输入：
1. 分词，将文本拆分为词元（Token）常见的分词方式包括字级、词级、子词级（如 BPE、WordPiece）、空格分词等。
    输入: "你好"
    分词: ["你", "好"]
2. 映射 将每个词元映射为词汇表中的唯一 ID，生成的数字序列即为模型的输入。
    分词: ["你", "好"]
    映射: [1001, 1002]
模型输出：
1. 返映射：模型输出的数字序列通过词汇表映射回对应的词元，二者是一一对应的关系。
    输出: [1001, 1002]
    反映射: ["你", "好"]
2. 文本重组：将解码后的词元以某种规则重新拼接为完整文本。
    反映射: ["你", "好"]
    重组: "你好"
"""

# 分词器类型
"""
BPE分词器：
示例：tokenizer = AutoTokenizer.from_pretrained("gpt2")

WordPiece分词器：使用 encode() 和 decode() 方法最简洁
示例：tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

"""

# 注意力掩码是什么？
'''
只关注实际的词元，忽略填充部分，避免无效的计算
- 1：表示模型应关注的词元（Tokens）
- 0：表示模型应忽略的词元（通常是填充 padding 的部分）。
'''


### 自定义处理数据集 -----------------------------------------------------------
class QA_Dataset(Dataset):
    """
    自定义的问答数据集类，用于处理问答任务的数据。

    参数：
    - split (str): 数据集的类型，'train'、'dev' 或 'test'。
    - questions (list): 问题列表，每个元素是一个字典，包含问题的详细信息。
    - tokenized_questions (BatchEncoding): 分词后的问题，由 tokenizer 生成。
    - tokenized_paragraphs (BatchEncoding): 分词后的段落列表，由 tokenizer 生成。
    
    属性（即 __init_() 中的 self.xxx）：
    - max_question_len (int): 问题的最大长度（以分词后的 token 数计）。
    - max_paragraph_len (int): 段落的最大长度（以分词后的 token 数计）。
    - doc_stride (int): 段落窗口滑动步长。
    - max_seq_len (int): 输入序列的最大长度。
    """
    
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 60
        self.max_paragraph_len = 150

        # 设置段落窗口滑动步长为段落最大长度的 10%
        self.doc_stride = int(self.max_paragraph_len * 0.1)

        # 输入序列长度 = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        """
        返回数据集中样本的数量。

        返回：
        - (int): 数据集的长度。
        """
        return len(self.questions)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的样本。

        参数：
        - idx (int): 样本的索引。

        返回：
        - 对于训练集，返回一个输入张量和对应的答案位置：
          (input_ids, token_type_ids, attention_mask, answer_start_token, answer_end_token)
        - 对于验证和测试集，返回包含多个窗口的输入张量列表：
          (input_ids_list, token_type_ids_list, attention_mask_list)
        """
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### 预处理 #####
        if self.split == "train":
            # 将答案在段落文本中的起始/结束位置转换为在分词后段落中的起始/结束位置
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # 防止模型学习到「答案总是位于中间的位置」，加入随机偏移
            mid = (answer_start_token + answer_end_token) // 2
            max_offset = self.max_paragraph_len // 2   # 最大偏移量为段落长度的1/2，这是可调的
            random_offset = np.random.randint(-max_offset, max_offset)  # 在 [-max_offset, +max_offset] 范围内随机选择偏移量
            paragraph_start = max(0, min(mid + random_offset - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # 切片问题/段落，并添加特殊标记（101：CLS，102：SEP）
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            # ... = [tokenizer.cls_token_id] + tokenized_question.ids[: self.max_question_len] + [tokenizer.sep_token_id]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]
            # ... = ... + [tokenizer.sep_token_id]

            # 将答案在分词后段落中的起始/结束位置转换为窗口中的起始/结束位置
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start

            # 填充序列，生成模型的输入
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        else:
            # 验证集和测试集的处理
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            # 段落被分割成多个窗口，每个窗口的起始位置由步长 "doc_stride" 分隔
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # 切片问题/段落并添加特殊标记（101：CLS，102：SEP）
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                # ... = [tokenizer.cls_token_id] + tokenized_question.ids[: self.max_question_len] + [tokenizer.sep_token_id]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                # ... = ... + [tokenizer.sep_token_id]

                # 填充序列，生成模型的输入
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        """
        对输入的序列进行填充，生成统一长度的模型输入。

        参数：
        - input_ids_question (list): 问题部分的输入 ID 列表。
        - input_ids_paragraph (list): 段落部分的输入 ID 列表。

        返回：
        - input_ids (list): 填充后的输入 ID 列表。
        - token_type_ids (list): 区分问题和段落的标记列表。
        - attention_mask (list): 注意力掩码列表，指示哪些位置是有效的输入。
        """
        # 计算需要填充的长度
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # 填充输入序列
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # 构造区分问题和段落的 token_type_ids
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # 构造注意力掩码，有效位置为 1，填充位置为 0
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len

        return input_ids, token_type_ids, attention_mask

# train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
# dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
# test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)


# 评估函数 ------------------------------------------------------------------------
def evaluate(data, output):
    """
    对模型的输出进行后处理，获取预测的答案文本。

    参数：
    - data (tuple): 包含输入数据的元组，(input_ids, token_type_ids, attention_mask)。
    - output (transformers.modeling_outputs.QuestionAnsweringModelOutput): 模型的输出结果。

    返回：
    - answer (str): 模型预测的答案文本。
    """
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # 通过选择最可能的起始位置/结束位置来获得答案
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # 确保起始位置索引小于或等于结束位置索引，避免选择错误的起始和结束位置对
        if start_index <= end_index:
            # 答案的概率计算为 start_prob 和 end_prob 的和
            prob = start_prob + end_prob

            # 如果当前窗口的答案具有更高的概率，则更新结果
            if prob > max_prob:
                max_prob = prob
                # 将标记转换为字符（例如，[1920, 7032] --> "大 金"）
                answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
        else:
            # 如果起始位置索引 > 结束位置索引，则跳过此对（可能是错误情况）
            continue
    # 移除答案中的空格（例如，"大 金" --> "大金"）
    return answer.replace(' ','')

# 训练部分 ---------------------------------------------------------------------------
# 超参数
'''
| 超参数                | 影响          | 调整建议                 |
| ------------------ | ----------- | -------------------- |
| `num_epoch`        | 训练轮次，影响学习深度 | 1\~5                 |
| `validation`       | 验证模型过拟合与否   | 一般设为 `True`          |
| `logging_step`     | 日志频率        | 50\~200              |
| `learning_rate`    | 学习速度        | 微调建议 `1e-5` 或 `2e-5` |
| `train_batch_size` | 训练稳定性与显存占用  | GPU 资源决定，建议 8\~32    |
'''
num_epoch = 1            # 训练的轮数，可以尝试修改
validation = True        # 是否在每个 epoch 结束后进行验证
logging_step = 100       # 每隔多少步打印一次训练日志
learning_rate = 1e-5     # 学习率
train_batch_size = 8     # 训练时的批次大小

# 优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 数据加载器
# 注意：不要更改 dev_loader / test_loader 的批次大小
# 虽然批次大小=1，但它实际上是由同一对 QA 的多个窗口组成的批次
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# 总训练步数
total_steps = len(train_loader) * num_epoch
num_warmup_steps = int(0.2 * total_steps)  # 这里设置 Warmup 步数为总步数的 20%

# [Hugging Face] 应用带有 warmup 的线性学习率衰减
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)

### 设置Accelerator 
#### 梯度累积（可选）####
# 注意：train_batch_size * gradient_accumulation_steps = 有效批次大小
# 如果 CUDA 内存不足，你可以降低 train_batch_size 并提高 gradient_accumulation_steps
# 文档：https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
gradient_accumulation_steps = 4

# 将 "fp16_training" 更改为 True 以支持自动混合精度训练（fp16）
fp16_training = True
if fp16_training:
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=gradient_accumulation_steps)
else:
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)


#开始训练
model.train()

print("开始训练...")

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0

    for data in tqdm(train_loader):
        with accelerator.accumulate(model):
            # 数据已经通过 accelerator.prepare() 移动到设备
            #data = [i.to(device) for i in data]

            # 模型输入：input_ids, token_type_ids, attention_mask, start_positions, end_positions（注意：只有 "input_ids" 是必需的）
            # 模型输出：start_logits, end_logits, loss（提供 start_positions/end_positions 时返回）
            output = model(
                input_ids=data[0],
                token_type_ids=data[1],
                attention_mask=data[2],
                start_positions=data[3],
                end_positions=data[4]
            )
            # 选择最可能的起始位置/结束位置
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # 只有当 start_index 和 end_index 都正确时，预测才正确
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()

            train_loss += output.loss

            accelerator.backward(output.loss)

            step += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 每经过 logging_step，打印训练损失和准确率（实际上，使用了梯度累积后的 step 应该除以对应的数量进行校正）
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

    if validation:
        print("评估开发集...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                # 这里保留了 device 的使用
                output = model(
                    input_ids=data[0].squeeze(0).to(device),
                    token_type_ids=data[1].squeeze(0).to(device),
                    attention_mask=data[2].squeeze(0).to(device),
                )
                # 只有当答案文本完全匹配时，预测才正确
                dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# 将模型及其配置文件保存到目录「saved_model」
# 即，在目录「saved_model」下有两个文件：「pytorch_model.bin」和「config.json」
# 可以使用「model = BertForQuestionAnswering.from_pretrained("saved_model")」重新加载保存的模型
print("保存模型...")
model_save_dir = "saved_model"
model.save_pretrained(model_save_dir)
#tokenizer.save_pretrained(model_save_dir)
