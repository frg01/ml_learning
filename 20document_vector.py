# 选择Transformers 和 LangChain 选择在 Retrieval -> Chinese 中表现较好的编码模型进行演示，即 chuxin-llm/Chuxin-Embedding。
'''
RAG (Retrieval-Augmented Generation 检索增强生成)： 在模型回答之前，先进行相关的信息提供给模型，已增强模型的准确性和相关性。
# 环境配置
pip install langchain langchain-community langchain-huggingface unstructured 
pip install pandas
pip install transformers sentence-transformers accelerate
pip install faiss-gpu
pip install optimum
pip install "numpy<2.0"

# 处理图片，tesseract 进行 OCR（以下为可选下载）
#sudo apt-get update
#sudo apt-get install python3-pil tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-script-latn
#pip install "unstructured[image]" tesseract tesseract-ocr
'''

# import nltk 

'''
- pubkt是无监督的句子边界检测算法，可将文本切分为句子，也可进行词语分割、
- punkt_tab是模型的辅助数据表或可能是内部开发测试内容
- averaged_perceptron_tagger 是平均感知机算法（Averaged Perceptron）的词性标注器（动词、名词、形容词等）
'''
# nltk.download('punkt')  # 下载 punkt 分词器
# nltk.download('punkt_tab')  # 下载 punkt_tab 分词器数据
# nltk.download('averaged_perceptron_tagger')  # 下载词性标注器
# nltk.download('averaged_perceptron_tagger_eng')  # 下载英文词性标注器

'''
步骤如下：
1. 使用预训练的编码器模型将「文档」内容编码为向量表示（embedding），然后建立一个向量数据库。
2. 在检索阶段，针对用户的「问题」，同样使用编码器将其编码为向量，然后在向量数据库中寻找与之相似的文档片段。
'''

# # 文档导入
# from langchain.document_loaders import DirectoryLoader 

# # 定义文件所在的路径
# DOC_PATH = "./Guide"

# # 使用DirectoryLoader 从指定路径加载文件。"*.md" 表示加载所有 .md 格式的文件，这里仅导入文章 10（避免当前文章的演示内容对结果的影响）
# loader = DirectoryLoader(DOC_PATH, glob="10*.md")

# # 加载目录中的指定的 .md 文件并将其转换为文档对象列表
# documents = loader.load()

# # 打印查看还在的文档内容
# print(documents[0].page_content[:200])


# text = """长隆广州世界嘉年华系列活动的长隆欢乐世界潮牌玩圣节隆重登场，在揭幕的第一天就吸引了大批年轻人前往打卡。据悉，这是长隆欢乐世界重金引进来自欧洲的12种巨型花车重磅出巡，让人宛若进入五彩缤纷的巨人国；全新的超级演艺广场每晚开启狂热的电音趴，将整个狂欢氛围推向高点。

# 记者在现场看到，明日之城、异次元界、南瓜欢乐小镇、暗黑城、魔域，五大风格迥异的“鬼”域在夜晚正式开启，全新重磅升级的十大“鬼”屋恭候着各位的到来，各式各样的“鬼”开始神出“鬼”没：明日之城中丧尸成群出行，寻找新鲜的“血肉”。异次元界异形生物游走，美丽冷艳之下暗藏危机。暗黑城亡灵出没，诅咒降临。魔域异“鬼”横行，上演“血腥恐怖”。南瓜欢乐小镇小丑当家，滑稽温馨带来欢笑。五大“鬼”域以灯光音效科技情景+氛围营造360°沉浸式异域次元界探险模式为前来狂欢的“鬼”友们献上“惊奇、恐怖、搞怪、欢乐”的玩圣体验。持续23天的长隆欢乐玩圣节将挑战游客的认知极限，让你大开眼界！
# 据介绍，今年长隆玩圣节与以往相比更为隆重，沉浸式场景营造惊悚氛围，两大新“鬼”王隆重登场，盛大的“鬼”王出巡仪式、数十种集声光乐和高科技于一体的街头表演、死亡巴士酷跑、南瓜欢乐小镇欢乐电音、暗黑城黑暗朋克、魔术舞台双煞魔舞、异形魔幻等一系列精彩节目无不让人拍手称奇、惊叹不止的“玩圣”盛宴让 “鬼”友们身临其境，过足“戏”瘾！
# """
# print(len(text))

# from langchain.text_splitter import RecursiveCharacterTextSplitter 
# # 创建一个文本分割器
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100, # 每个文本块的最大长度
#     chunk_overlap=0, # 文本块之间的字符重叠数量
#     separators=["\n\n","\n",",","。",""],
#     # length_function=lambda x: 1, # lambda text: len(tokenizer.encode(text)) 这样是按照token数分割
# )

# # 将文本分割成多个块
# texts = text_splitter.split_text(text)

# # 打印分割后的文本块数量
# print(len(texts))

# # 打印第一个文本块的长度
# print(len(texts[0]))

# # 打印第一个文本块的最后20个字符
# print(texts[0][80:])

# # 打印第二个文本块的前20个字符
# print(texts[1][:20])

# for i,t in enumerate(texts):
#     print(f"Chunk{i+1} length: {len(t)}")
#     print(t)
#     print("-" * 50)

### 加载编码模型 -----------------------------------------------------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings 

# 指定要加载的预训练模型的任务， 参考排行榜https://huggingface.co/spaces/mteb/leaderboard
model_name = "chuxin-llm/Chuxin-Embedding"

# 创建 Hugging Face 的嵌入模型实例，这个模型将用于将文本转换为向量表示（embedding）
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# 打印嵌入模型的配置信息， 显示模型结构和其他相关参数
print(embedding_model)

# embed_query() 方法将文本转换为嵌入的向量
query_embedding = embedding_model.embed_query("Hello")

# 打印生成的嵌入向量的长度， 向量长度应与模型的输出维度一致（这里是 1024），你也可以选择打印向量看看
print(f"嵌入向量的维度为: {len(query_embedding)}")


### 保存和加载向量数据库
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

CONNECTION_STRING = "postgresql+psycopg2://hwc@localhost:5432/test3"
COLLECTION_NAME = "state_of_the_union_test"
embeddings = OpenAIEmbeddings()
'''
CREATE TABLE my_collection (
    id uuid PRIMARY KEY,
    embedding vector(768), -- 向量列（dim = embedding 模型输出维度）
    document text,          -- 原始文本
    metadata jsonb          -- 可选，文档元信息
);
'''
docs = [
    Document(page_content="这是第一段文本", metadata={"source": "file1.txt"}),
    Document(page_content="这是第二段内容", metadata={"source": "file2.txt"}),
]

vectordb = PGVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="my_collection",
    connection_string=CONNECTION_STRING,
    use_jsonb=True  # 推荐开启，meta存成jsonb格式
)


vectordb = PGVector(
    embedding_function=embedding_model,
    collection_name="your_collection",
    connection_string=CONNECTION_STRING
)

# 变成检索器
retriever = vectordb.as_retriever()

# 插入文档和向量
vectordb.add_embeddings(
    texts=[doc.page_content for doc in docs],
    embeddings=embeddings,
    metadatas=[doc.metadata for doc in docs]
)

# 进行相似检索
query = "介绍一下生成式大语言模型的采样策略"
results = vectordb.similarity_search(query, k=3)

for i, result in enumerate(results):
    print(f"Result #{i+1}")
    print("内容：", result.page_content)
    print("元数据：", result.metadata)
    print("=" * 40)



### 进行一个文本生成的任务 -------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

# 以下二选一，也可以进行替换
# 本地
model_path = './Mistral-7B-Instruct-v0.3-GPTQ-4bit'
# 远程
model_path = 'neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit'

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",  # 自动选择模型的权重数据类型
    device_map="auto",   # 自动选择可用的设备（CPU/GPU）
)

# 创建transformers 的 管道
from transformers import pipline 
generator = pipline(
    "text-generation", # 指定任务类型为文本生成、
    model=model,
    tokenizer=tokenizer,
    max_length=4096,   # 指定生成文本的最大长度
    path_token_id=tokenizer.eos_token_id,
)

# 集成到LangChain
from langchain_huggingface import HuggingFacePipeline  
llm = HuggingFacePipeline(pipline=generator)

# 定义提示词模板
from langchain.prompts import PromptTemplate 
custom_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)


#构建问答链
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 直接堆叠所有检索到的文档
    retriever=retriever, # 使用先前定义的检索器来获取相关文档
    # chain_type_kwargs={"prompt": custom_prompt}  # 可以选择传入自定义提示模板（传入的话记得取消注释），如果不需要可以删除这个参数
)

# 提出问题
query = "Top-K 和 Top-P 的区别是什么？"

# 获取答案
'''
# 表示对chain进行调用，把query传入回答链
自动完成检索 → 组装Prompt → 调用LLM → 生成回答 这一整套流程。
invoke 1. 调用Retriever. 2. 构造Prompt 3. 调用LLM 4. 返回回答 
'''
answer = qa_chain.invoke(query) 
# print(answer)  # 可以对比 qa_chain.run() 和 qa_chain.invoke() 在返回上的差异
print(answer['result'])


### 完整构造RAG -------------------------------------------------------------------------------
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# 定义文件所在的路径
DOC_PATH = "../Guide"

# 使用DirectoryLoader 从指定路径加载文件。"*.md" 表示加载所有 .md 格式的文件，这里仅导入文章 10（避免文章 20 的演示内容对结果的影响）
loader = DirectoryLoader(DOC_PATH, glob="10*.md")

# 加载目录中的指定的 .md 文件并将其转换为文档对象列表
documents = loader.load()

# 文本处理
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 尝试调整它
    chunk_overlap=100,  # 尝试调整它
    #length_function=len,  # 可以省略
    #separators=["\n\n", "\n", " ", "。", ""]  # 可以省略
)
docs = text_splitter.split_documents(documents)


# 生成嵌入（使用 Hugging Face 模型）
# 指定要加载的预训练模型的名称，参考排行榜：https://huggingface.co/spaces/mteb/leaderboard
model_name = "chuxin-llm/Chuxin-Embedding"

# 创建 Hugging Face 的嵌入模型实例，这个模型将用于将文本转换为向量表示（embedding）
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# 建立向量数据库
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    collection_name="my_collection",
    connection_string=CONNECTION_STRING,
    use_jsonb=True  # 推荐开启，meta存成jsonb格式
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 加载文本生成模型
# 本地
model_path = './Mistral-7B-Instruct-v0.3-GPTQ-4bit'
# 远程
#model_path = 'neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit'

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",  # 自动选择模型的权重数据类型
    device_map="auto",   # 自动选择可用的设备（CPU/GPU）
)

# 创建文本生成管道
generator = pipeline(
    "text-generation",  # 指定任务类型为文本生成
    model=model,
    tokenizer=tokenizer,
    max_length=4096,    # 指定生成文本的最大长度
    pad_token_id=tokenizer.eos_token_id
)

# 包装为langchain 的LLM 接口
llm = HuggingFacePipeline(pipeline=generator)

custom_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

# 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 直接堆叠所有检索到的文档
    retriever=retriever, # 使用先前定义的检索器来获取相关文档
    # chain_type_kwargs={"prompt": custom_prompt}  # 可以选择传入自定义提示模板（传入的话记得取消注释），如果不需要可以删除这个参数
)

# 提出问题
query = "Top-K 和 Top-P 的区别是什么？"

# 获取答案
answer = qa_chain.invoke(query)
print(answer['result'])





### 扩展延伸 split_text的解析-------------------------------------------
'''
def _split_text(text, separators):
    → 1. 找到当前合适的分隔符（从列表中优先使用能在当前 text 中找到的）
    → 2. 用这个分隔符把 text 拆成多个 split 块
    → 3. 遍历每个 split：
        - 若它长度小于 chunk_size，则缓存起来
        - 若它过长：
            → 若还有更多分隔符可选，则递归调用 _split_text 继续拆分
            → 否则直接添加到最终结果中
    → 4. 把缓存的 split 合并成最终 chunks
    → 5. 返回 chunks 列表

["\n\n", "\n", ".", ""]
拆分器会优先尝试按段落（\n\n）拆，拆不了就尝试句子（\n）、句号（.），直到按字符级别（""）来拆。这样就能 最大限度保持语义完整性，同时满足大小限制。

'''
