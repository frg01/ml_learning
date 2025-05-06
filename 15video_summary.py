#  AI视频摘要的本质是，视频 -> 音频 -> 文本  ->  LLMs总结

# 准备工作，视频转音频 ---------------------------------------------------------------
'''
sudo apt-get install ffmpeg

# -q:a 0：设置音频质量，0 为最高音频质量。-map a：只提取音频部分
ffmpeg -i input_video.mp4 -q:a 0 -map a output_audio.mp3

安装对应的库
pip install srt
pip install datasets
pip install DateTime
pip install OpenCC==1.1.6
pip install opencv-contrib-python
pip install opencv-python
pip install opencv-python-headless
pip install openpyxl
pip install openai
pip install git+https://github.com/openai/whisper.git@ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab
pip install numpy
pip install soundfile
pip install ipywidgets
pip install librosa
'''

import os, time, re, pathlib, textwrap, datetime
# 第三方库
import numpy as np
import srt 
import soundfile as sf
from tqdm import tqdm

# 项目相关库
import whisper 
from datasets import load_dataset
from openai import OpenAI

# 加载本地 Parquet 格式的数据集
# dataset = load_dataset("kuanhuggingface/NTU-GenAI-2024-HW9")
# # del(ds)
dataset = load_dataset('parquet', data_files={'test': './Demos/data/13/test-00000-of-00001.parquet'})

# 准备音频
input_audio = dataset["test"]["audio"][0]
input_audio_name = input_audio["path"]                        #音频文件名
input_audio_array = input_audio["array"].astype(np.float32)   #音频数据
sampling_rate = input_audio["sampling_rate"]                  # 采样率

# 加载.mp3 文件 Demos/data/13/audio.mp3 的文件为例：
import librosa 
# 指定 MP3 文件路径
mp3_file_path = './Demos/data/13/audio.mp3'
'''
input_audio_array：音频数据的 NumPy 数组表示。
sampling_rate：音频的采样率（Hz）。
input_audio_name：音频文件名，仅保留文件名，不包含路径。
注意：我们使用 sr=16000，将音频采样率转换为 Whisper 模型要求的 16000 Hz，确保模型能够正确处理音频数据。
'''

input_audio_name = os.path.basename(mp3_file_path)

# 加载音频文件，指定采样率为 16000
input_audio_array, sampling_rate = librosa.load(mp3_file_path, sr=16000)

# 打印音频数据的采样率和形状，确保加载成功
print(f"采样率: {sampling_rate}")
print(f"音频数据形状: {input_audio_array.shape}")

print(f"现在我们将转录音频: ({input_audio_name})。")

### 自动语音识别 --------------------------------------------------------------------------------

# 定义语音识别函数
def speech_recognition(model_name, input_audio, output_subtitle_path, decode_options, cache_dir="./"):
    # 加载模型
    model = whisper.load_model(name=model_name, download_root=cache_dir)

    # 转录音频
    transcription = model.transcribe(
        audio=input_audio,
        language=decode_options['language'],
        verbose=False,
        initial_prompt=decode_options["initial_prompt"],
        temperature=decode_options["temperature"]
    )

    # 处理转录结果， 生成字幕文件
    subtitles = []
    for i, segment in enumerate(transcription["segments"]):
        start_time = datetime.timedelta(seconds=segment["start"])
        end_time = datetime.timedelta(seconds=segment["end"])
        text = segment["text"]
        subtitles.append(srt.Subtitle(index=i, start=start_time, end=end_time, content=text))
    
    srt_content = srt.compose(subtitles)

    # 保存字幕文件
    with open(output_subtitle_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    print(f"字幕已保存到{output_subtitle_path}")

### 设置模型参数 运行语音识别
# 模型名称，可选 'tiny', 'base', 'small', 'medium', 'large-v3'
model_name = 'medium'

# 语言
language = 'zh'  # 选择语音识别的目标语言，如 'zh' 表示中文

# 初始 prompt，可选
initial_prompt = '请用中文'  # 如果需要，可以为 Whisper 模型设置初始 prompt 语句

# 采样温度，控制模型的输出多样性
temperature = 0.0  # 0 表示最确定性的输出，范围为 0-1

# 输出文件后缀
suffix = '信号与人生'

# 字幕文件路径
output_subtitle_path = f"./output-{suffix}.srt"

# 模型缓存目录
cache_dir = './'

# 运行语音识别
decode_options = {
    "language": language,
    "initial_prompt": initial_prompt,
    "temperature": temperature
}

# 运行ASR
speech_recognition(
    model_name=model_name,
    input_audio=input_audio_array,
    output_subtitle_path=output_subtitle_path,
    decode_options=decode_options,
    cache_dir=cache_dir
)

# 读取并打印字幕内容
with open(output_subtitle_path, 'r', encoding='utf-8') as file:
    content = file.read()
print(content)



### 处理语音识别的字幕
def extract_and_save_text(srt_filename, output_filename):
    #读取SRT 文件
    with open(srt_filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # 去除时间戳和索引
    pure_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
    pure_text = re.sub(r'\n\n+', '\n', pure_text)

    # 保存纯文本
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(pure_text)

    print(f'提取的文本已保存到 {output_filename}')

    return pure_text

# 拆分文本
def chunk_text(text, max_length):
    return textwrap.wrap(text, max_length)

# 文本块长度
chunk_length = 512

# 提取文本并拆分
pure_text = extract_and_save_text(
    srt_filename=output_subtitle_path,
    output_filename=f"./output-{suffix}.txt",
)

chunks = chunk_text(text=pure_text, max_length=chunk_length)


### 文本摘要 分段和精炼--------------------------------------------------------------------
# 模型名称
model_name = 'qwen-turbo'

# 控制响应的随机性
temperature = 0.0

# 控制多样性
top_p = 1.0

# 最大生成标记数
max_tokens = 512


# 不设置则默认使用环境变量
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 构建OpenAI 客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
)

# 模型名称
model_name = 'qwen-turbo'

# 控制响应的随机性
temperature = 0.0

# 控制多样性
top_p = 1.0

# 最大生成标记数
max_tokens = 512

# 定义摘要函数
def summarization(client, summarization_prompt, model_name="qwen-turbo", temperature=0.0, top_p=1.0, max_tokens=512):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": summarization_prompt}],
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# 定义摘要提示模板
summarization_prompt_template = "尽量凝练内容，但要保证包括要点和所有重要细节：<text>"

# 对每个文本块生成摘要
paragraph_summaries = []
for index, chunk in enumerate(chunks):
    print(f"\n========== 正在生成第 {index + 1} 段摘要 ==========\n")
    print(f"原始文本 (第 {index + 1} 段):\n{chunk}\n")
    
    # 构建摘要提示
    summarization_prompt = summarization_prompt_template.replace("<text>", chunk)
    
    # 调用摘要函数
    summary = summarization(
        client=client,
        summarization_prompt=summarization_prompt,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # 打印生成的摘要
    print(f"生成的摘要 (第 {index + 1} 段):\n{summary}\n")
    
    # 将生成的摘要保存到列表
    paragraph_summaries.append(summary)

# 合并段落摘要
collected_summaries = "\n".join(paragraph_summaries)

# 定义最终摘要提示模板
final_summarization_prompt = "在 500 字以内写出以下文字的简洁摘要：<text>"
final_summarization_prompt = final_summarization_prompt.replace("<text>", collected_summaries)

# 生成最终摘要
final_summary = summarization(
    client=client,
    summarization_prompt=final_summarization_prompt,
    model_name=model_name,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens
)

print(final_summary)
