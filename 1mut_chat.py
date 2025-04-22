from openai import OpenAI
import os

os.environ['OPENAI_API_KEY'] = "sk-738af55c0b224bf79edca4b48a95dab1"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY',"null")
assert OPENAI_API_KEY != "null"

# 初始化OpenAI客户端
client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

message = [{'role':'system','content':'you are a helpful assistant.'},
           {'role':'user','content':'你是谁？'}]

# 单论对话
# try:   
#     completion = client.chat.completions.create(
#         model="qwen-turbo",
#         messages=[{'role':'system','content':'you are a helpful assistant.'},
#            {'role':'user','content':'你是谁？'}]
#     )
#     print(completion.choices[0].message.content)

# except Exception as e:
#     print(f"错误信息：{e}")


# # 多轮对话
# messages = [{"role": "system","content":"你是一个帮助助手。"}]
# try:
#     for i in range(3):
#         #输入问题
#         user_input = input("请输入：")

#         #添加用户消息到对话历史
#         messages.append({"role":"user","content":user_input})

#         response = client.chat.completions.create(
#             model="qwen-turbo",
#             messages=messages
#         )

#         assistant_output = response.choices[0].message.content
#         messages.append({"role":"assistant","content":assistant_output})

#         print(f'用户输入：{user_input}')
#         print(f'模型输出：{assistant_output}')
# except Exception as e:
#     print(f"错误信息：{e}")


# 开启流式输出,实时了解到生成信息，先对输出内容进行阅读
try:
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {'role':'system','content':'你是一个帮助助手'},
            {'role':'user','content':'请你介绍我web3'}
        ],
        stream=True,
    )

    #实时打印模型回复的增量内容
    for chunk in response:
        #判断回复内容是否为空
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content,end='')
except Exception as e:
    print("错误内容：{e}")