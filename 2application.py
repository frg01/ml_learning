import os 
import json 
from typing import List,Dict,Tuple

import openai
import gradio as gr

os.environ['OPENAI_API_KEY'] = "sk-738af55c0b224bf79edca4b48a95dab1"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY',"null")
assert OPENAI_API_KEY != "null"

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

try:
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{'role':'user','content':'测试'}],
        max_tokens=1,
    )
    print("API设置成功！！")

except Exception as e:
    print(f"API 可能有问题，请检查：{e}")


# #part1 文章摘要
# prompt="请你帮我写一下文章的摘要"
# arcticle="太史公曰：《诗》有之：“高山仰止，景行行止。”虽不能至，然心乡往之。余读孔氏书，想见其为人。适鲁，观仲尼庙堂车服礼器，诸生以时习礼其家，余祗回留之不能去云。天下君王至于贤人众矣，当时则荣，没则已焉。孔子布衣，传十余世，学者宗之。自天子王侯，中国言《六艺》者折中于夫子，可谓至圣矣！"
# try:
#     input_text=f"{prompt}\n{arcticle}"

#     response = client.chat.completions.create(
#         model="qwen-turbo",
#         messages=[{'role': 'user','content': input_text}]
#     )

#     print(response.choices[0].message.content)
# except Exception as e:
#     print(f"报错信息：{e}")

# PROMPT_FOR_SUMMARIZATION = "请将以下文章概括成几句话。"

def reset():
    """
    清空对话记录
    返回：
        List： 空的对话记录列表
    """
    return []

def interact_summarization(prompt, arcticle, temp=1.0):
    """
    调用模型生成摘要。

    参数:
        prompt (str): 用于摘要的提示词
        article (str): 需要摘要的文章内容
        temp (float): 模型温度，控制输出创造性（默认 1.0）

    返回:
        List[Tuple[str, str]]: 对话记录，包含输入文本与模型输出
    """
    # 合成请求文本
    input_text = f"{prompt}\n{arcticle}"

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{'role': 'user', 'content':input_text}],
        temperature=temp
    )
    
    return [(input_text,response.choices[0].message.content)]


def export_summarization(chatbot, article):
    """
    导出摘要任务的对话记录和文章内容到 JSON 文件。

    参数:
        chatbot (List[Tuple[str, str]]): 模型对话记录
        article (str): 文章内容
    """
    target = {"chatbot":chatbot, "article":article}
    with open("files/part1.json","w",encoding='utf-8') as file:
        json.dump(target,file,ensure_ascii=False,indent=4)

# # 构建 Gradio UI 界面
# with gr.Blocks() as demo:
#     gr.Markdown("# 第1部分：摘要\n填写任何你喜欢的文章，让聊天机器人为你总结！")
#     chatbot = gr.Chatbot()
#     prompt_textbox = gr.Textbox(label="提示词", value=PROMPT_FOR_SUMMARIZATION, visible=False)
#     article_textbox = gr.Textbox(label="文章", interactive=True, value="填充")
    
#     with gr.Column():
#         gr.Markdown("# 温度调节\n温度用于控制聊天机器人的输出，温度越高，响应越具创造性。")
#         temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")
    
#     with gr.Row():
#         send_button = gr.Button(value="发送")
#         reset_button = gr.Button(value="重置")
    
#     with gr.Column():
#         gr.Markdown("# 保存结果\n当你对结果满意后，点击导出按钮保存结果。")
#         export_button = gr.Button(value="导出")
    
#     # 绑定按钮与回调函数
#     send_button.click(interact_summarization,
#                       inputs=[prompt_textbox, article_textbox, temperature_slider],
#                       outputs=[chatbot])
#     reset_button.click(reset, outputs=[chatbot])
#     export_button.click(export_summarization, inputs=[chatbot, article_textbox])


# # 启动 Gradio 应用
# demo.launch(debug=True)



# part2 角色扮演
PROMPT_FOR_ROLEPLAY="我需要你面试我有关AI的知识，仅提出问题" 

#预设角色
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[{'role': 'user', 'content': PROMPT_FOR_ROLEPLAY}],
)

assistant_reply = response.choices[0].message.content

messages = []
messages.append({'role': 'user','content':PROMPT_FOR_ROLEPLAY})
messages.append({'role': 'assistant', 'content': assistant_reply})

#进行三轮对话
for _ in range(3):
    user_input = input("请输入：")
    messages.append({'role':'user','content':user_input})

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content
    messages.append({'role':'assistant','content':assistant_reply})

    print(f"用户输入：{user_input}")
    print(f"模型回复：{assistant_reply}\n")


# Gradio 可视化
CHARACTER_FOR_CHATBOT="面试官"
PROMPT_FOR_ROLEPLAY="我需要你面试我有关AI的知识，仅提出问题" 
assistant_reply = response.choices[0].message.content

def reset():
    """
    清空对话记录，返回：List空的对话记录列表
    """
    return []

def interact_roleplay(chatbot,user_input,temp=1.0):
    """
    处理角色扮演多轮对话，调用模型生成回复。

    参数:
        chatbot (List[Tuple[str, str]]): 对话历史记录（用户与模型回复）
        user_input (str): 当前用户输入
        temp (float): 模型温度参数（默认 1.0）

    返回:
        List[Tuple[str, str]]: 更新后的对话记录
    """
    try:
        messages =[]
        for input_text,response_text in chatbot:
            messages.append({'role': 'user', 'content': input_text})
            messages.append({'role': 'assistant', 'content': response_text})
        
        messages.append({'role': 'user', 'content': user_input})

        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            temperature=temp,
        )

        chatbot.append((user_input,response.choices[0].message.content))
    except Exception as e:
        print(f"错误信息：{e}")
        chatbot.append((user_input,f"报错，发生了错误：{e}"))
    
    return chatbot


def export_roleplay(chatbot,description):
    """
    导出角色扮演对话记录及任务描述到 JSON 文件。

    参数:
        chatbot (List[Tuple[str, str]]): 对话记录
        description (str): 任务描述
    """
    target = {"chatbot":chatbot,"description": description}
    with open("files/part2.json","w",encoding="utf-8") as file:
        json.dump(target,file,ensure_ascii=False,indent=4)


#进行第一次对话：设定角色提示
first_dialogue = interact_summarization([],PROMPT_FOR_ROLEPLAY)

#构建Gradio UI界面
with gr.Blocks() as demo:
    gr.Markdown("# 第2部分：角色扮演\n与聊天机器人进行角色扮演互动！")
    chatbot = gr.Chatbot(value=first_dialogue)
    description_textbox = gr.Textbox(label="机器人扮演的角色", interactive=False, value=CHARACTER_FOR_CHATBOT)
    input_textbox = gr.Textbox(label="输入", value="")
    
    with gr.Column():
        gr.Markdown("# 温度调节\n温度控制聊天机器人的响应创造性。")
        temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")
    
    with gr.Row():
        send_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")
    
    with gr.Column():
        gr.Markdown("# 保存结果\n点击导出按钮保存对话记录。")
        export_button = gr.Button(value="导出")
        
    # 绑定按钮与回调函数
    send_button.click(interact_roleplay, inputs=[chatbot, input_textbox, temperature_slider], outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_roleplay, inputs=[chatbot, description_textbox])
    
# 启动 Gradio 应用
demo.launch(debug=True)