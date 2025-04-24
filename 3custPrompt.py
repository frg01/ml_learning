import os
import time
import re
import pickle
import json
import traceback

import openai
import tiktoken  # 用于 prompt_token_num()
import jinja2
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = "sk-738af55c0b224bf79edca4b48a95dab1"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY',"null")
assert OPENAI_API_KEY != "null"

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#检查API设置是否正确
try:
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": "测试!"}],
    )
    print("API设置正确。")
except Exception as e:
    print(f"API错误信息: {e}")


#初始化OpenAI客户端
class OpenAIModel:
    """
    封装OpenAI API调用和缓存机制的类

    用于调用OpenAI API，处理响应，并缓存结果以提高效率

    属性：
        cache_file (str): 缓存文件的路径
        cache_dict (dict): 内存中的缓存字典
    """

    def __init__(self, cache_file='openai_cache'):
        """
        初始化OpenAIModel模型对象，设置缓存文件路径并加载缓存 

        参数：
            cache_file (str): 缓存文件的路径，默认为'openai_cache'
        """
        self.cache_file = cache_file
        self.cache_dict = self.load_cache() #加载缓存

    def save_cache(self):
        """
        将当前缓存保存到文件中
        """
        with open(self.cache_file,'wb') as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self,allow_retry=True):
        """
        从文件加载缓存，支持重试机制。

        参数：
            allow_retry (bool): 是否允许重试加载缓存，默认为True
        返回：
            dict: 加载的缓存字典,如果文件不存在则返回空字典
        """
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, 'rb') as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print("Pickle Error, 5秒后重试")
                    time.sleep(5)
        else:
            #如果文件不存在则初始化缓存
            cache = {}
        return cache

    def set_cache(self,file_name):
        """
        设置缓存文件名并重新加载缓存
        参数:
            file_name (str): 缓存文件的路径
        """
        self.cache_file = file_name
        self.cache_dict = self.load_cache()

    def get_response(self,content):
        """
        获取模型完成的文本，先检查缓存，若无则请求生成
        参数：
            content (str): 提供给模型的输入内容
        返回：
            str: 模型生成的回复文本，如果出错则返回错误信息
        
        """
        #如果选择检查缓存，则会导致同问题不同trilal的结果相同，这与想表达的内容不符，故注释
        # if content in self.cache_dict:
        #     return self.cache_dict[content]
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[{"role": "user", "content": content}],
                    temperature=1,
                )
                
                completion = response.choices[0].message.content
                return completion
            except Exception as e:
                print(e,"\n")
                time.sleep(1)

        return None


    def is_valid_key(self):
        """
        检查API密钥是否有效

        返回：
            bool: 如果密钥有效返回True，否则返回False
        """
        for _ in range(4):
            try:
                response = client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[{"role": "user", "content": "测试!"}],
                    temperature=1,
                    max_tokens=1,
                )
                return True
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)

        return False
    

    def prompt_token_num(self,prompt):
        """
        计算prompt的token数量
        
        参数：
            prompt (str): 要计算token数量的prompt
            
        返回:
            int: token的数量
        """

        try:
            #使用gpt-3.5-turbo的编码器，因为tiktoken库不支持
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4兼容 tokenizer
            #将prompt编码成token，并返回token的数量
            tokens = encoding.encode(prompt)
            return len(tokens)
        except Exception as e:
            print(f"Error in prompt_token_num: {e}")
            return 0
        
    def two_stage_completion(self,question,content):
        """
        两阶段完成：首先获取推理，直接输入content，然后将rationale和question获取问题的答案
        参数：
            question (str): 原始问题
            content (str): 提供给模型的输入内容
        返回：
            dict: 包含prompt、推理和答案的字典
        """
        rationale = self.get_response(content)
        if not rationale:
            return {
                "prompt":content,
                "rationale": None,
                "answer": None,
            }
        ans = self.get_response(content=f"Q:{question}\nA:{rationale}") 
        return {
            "prompt":content,
            "rationale": rationale,
            "answer": ans,
        }
#初始化模型
my_model = OpenAIModel()


# 评估prompt的数学问题
questions = [
    '一位艺术家正在使用方形瓷砖创建一个大型马赛克。马赛克本身设计成一个正方形，并且包含的蓝色瓷砖数量恰好是红色瓷砖的3倍。如果艺术家只有57块红色瓷砖，那么要完成整个马赛克共需要多少块瓷砖？',
    '一位农民正在为当地市场装苹果。他有120个苹果，并希望将它们均匀分配到篮子中。如果他决定留15个苹果给家人，每个篮子最多能装7个苹果，那么他最少需要多少个篮子才能将苹果带到市场？',
    '一个花园有矩形的地块，这些地块排列成一条直线，每块地与正好两块其他的地共用边界。共有5块地。中间的每块地面积为24平方米，地块的宽度为4米，所有地块的宽度保持不变。第一块地的长度是中间地块的两倍，最后一块地的长度是中间地块的一半。那么所有地块的总面积是多少平方米？',
    '一个农贸市场出售两种类型的苹果混合袋：A型袋子包含4个红苹果和6个绿苹果，B型袋子包含8个红苹果和4个绿苹果。一位顾客购买了一袋A型和一袋B型的苹果。如果从这两袋苹果中随机挑选一个苹果，选到绿苹果的概率是多少？请将答案保留到小数点后两位。',
    '一位园丁按照两朵红色花跟着一朵黄色花的模式种花。如果园丁想保持这种模式，并且有35个连续的空位来种花，那么园丁会种多少朵红色花？',
    '杰森正在为马拉松训练，他每天跑固定的距离。星期一，他跑了5英里。之后的每一天，他将跑步距离增加10%。如果杰森按照这个模式继续跑步，那么他在星期五将跑多少英里？',
    '在一个三角形的花坛边上，每条边上有16棵植物。每棵植物都需要一个半径为0.5米的圆形空间才能正常生长。假设植物紧挨着排列，并且沿着三角形花坛的边排成一条直线，那么每条边上种植物的线性距离是多少米？',
    '威尔逊博士正在设计一个几何花园，花园中的花朵围绕着中央的喷泉排列成同心圆。每一圈比里面一圈多6朵花，形成一个六边形的图案。最里面一圈有6朵花。如果威尔逊博士种足够的花，形成15圈（包括最里面一圈），那么这个花园总共需要多少朵花？',
    '一个小图书馆正在重新整理书籍收藏。他们总共有120本书，计划平均分配到5个书架上。最上面的书架只能承受相当于最下面书架一半重量的书。如果最上面的书架承载15磅的书，而其他书架每个能承载两倍的重量，那么最下面的书架能承载多少磅的书？',
    '一份饼干的配方需要3杯面粉、2杯糖和1杯巧克力片。如果马克想要做三倍量的饼干，但只有4杯糖，那么他还需要多少杯糖？',
    '一家宠物店的店主正在制作定制鸟舍。每个鸟舍外部需要0.75升木材清漆。如果店主有一罐10升的木材清漆，那么他在需要更多清漆之前最多可以制作多少个鸟舍？',
    '一个农场有鸡和牛。总共有30个头，88条腿。农场上有多少头牛？',
    '一个地方图书馆正在组织一场旧书义卖会，以筹集资金购买新书。他们以每本2美元的价格卖出120本儿童书，以每本3美元的价格卖出75本小说，并以每本1.50美元的价格卖出了小说两倍数量的杂志。他们还以每本0.50美元的价格卖出与书籍和杂志总数相等的书签。那么图书馆总共筹集了多少钱？',
    '一个当地的农民正在为市场准备混合水果篮，每个篮子包含3个苹果、5个橙子和2个香蕉。苹果的价格是每个0.50美元，橙子每个0.30美元，香蕉每个0.25美元。如果农民为当地市场准备了120个篮子，并以每个5.00美元的价格出售每个篮子，那么卖完所有篮子后，农民将获得多少利润？',
    '玛丽亚有24个苹果，想将它们均匀分给她的6个朋友。如果每个朋友还要再给老师2个苹果，那么每个朋友剩下多少苹果？',
    '莉拉正在计划一个花园，想要种三种花：雏菊、郁金香和玫瑰。她想要的雏菊数量是郁金香的两倍，郁金香的数量是玫瑰的三倍。如果她总共要种60朵花，那么她计划种多少朵玫瑰？',
    '一个花园有三种开花植物。第一种每株有12朵花，第二种每株有8朵花，第三种每株有15朵花。如果第一种植物的数量是第二种植物的两倍，第三种植物的数量是第一种植物的一半，并且花园中有16株第二种植物，那么花园里一共有多少朵花？',
    '在一个棋盘游戏中，从一个方格转移到另一个方格的费用是你要落在的方格号码的硬币数。第一个方格是1号，第二个方格是2号，以此类推。如果一个玩家从5号方格移动到9号方格，再到14号方格，最后到20号方格，他总共花费了多少枚硬币？',
    '一个景观公司在两个公园种植树木。在A公园，他们种了5排，每排6棵树。在B公园，他们种了3排，每排7棵树。然而，B公园的4棵树没有成活，必须移除。移除之后，总共剩下多少棵树？',
    '欧拉博士正在计划一场数学比赛，他决定将参与者分成几组。为了保证公平，每组必须有相同数量的参与者。如果欧拉博士可以选择将参与者分成4人、5人或6人的组，并且参与者总数少于100人，那么他最多可以有多少参与者，确保无论怎么分组都不会有剩余？',
    '一个农民为万圣节种植南瓜。他种了8排，每排15棵南瓜植株。每棵植株平均产出3个南瓜。收获后，农民将20%的南瓜卖给当地市场，剩下的在他的农场摊位上出售。如果每个南瓜卖4美元，农民通过销售南瓜总共赚了多少钱？',
    '一个三角形公园ABC的边缘上种植了树木。边AB上的树木数量等于边BC的长度，边BC上的树木数量等于边CA的长度，边CA上的树木数量等于边AB的长度。如果边AB、BC和CA（以米为单位）的长度构成一个公比为2的几何级数，并且总共种植了77棵树，求边AB的长度。',
    '一群朋友正在收集可回收的罐子。玛雅收集的罐子是利亚姆的两倍。利亚姆收集了15个罐子。如果佐伊比玛雅多收集了5个罐子，并且这群朋友想把罐子平分给4家慈善机构，每家会收到多少个罐子？',
    '在一场科学比赛中，每个团队需要制作一个模型火箭。有6个团队，每个团队需要一套材料。材料包括火箭的主体管、引擎和降落伞。主体管每个12.50美元，引擎每个18.75美元，降落伞每个6.25美元。购买所有团队的材料后，总费用为225美元。制作一支火箭的材料费用是多少？',
    '艾米丽有一个小菜园，种植了番茄、胡萝卜和黄瓜。她的番茄植株数量是黄瓜植株的两倍，而胡萝卜植株比番茄少5棵。如果艾米丽有4棵黄瓜植株，那么她总共有多少棵菜园植物？',
    '在一个小村庄，当地裁缝制作外套和裤子。制作一件外套需要3码布料，而制作一条裤子需要2码布料。他接到了一份剧院制作的订单，要求的裤子数量是外套的两倍，而剧院要求了4件外套。如果布料的价格是每码15美元，那么剧院在这个订单上需要花费多少布料费用？',
    '一个小镇的人口以恒定的速率增长。如果2010年小镇的人口是5000人，2020年是8000人，那么如果这种增长趋势继续，到2025年小镇的人口会是多少？',
    '一位数学老师正在组织一场测验比赛，并决定用铅笔作为奖品。每位参与者将获得2支铅笔，而得分超过80%的学生将额外获得3支铅笔。如果班上有30名学生，其中1/5的学生得分超过80%，那么老师需要准备多少支铅笔？',
    '一个长方形的花园被120米的围栏包围。如果花园的长度是其宽度的三倍，那么花园的面积是多少平方米？',
    '一个长10米、宽15米的花园将用方形瓷砖铺设。每块瓷砖的边长为25厘米。如果每块瓷砖的价格是3美元，而铺设瓷砖的人工费用是每平方米8美元，那么铺设整个花园的总费用是多少？'
]
answers = [
    228, 15, 132, 0.45, 24, 7.3205, 16, 720, 30, 2, 13, 14, 862.5, 180, 2, 6, 752, 
    43, 47, 60, 1440, 11, 20, 37.5, 15, 420, 9500, 78, 675, 8400
]


# 创建自定义Prompt（Gradio版本）
import gradio as gr 

def reset_prompt(chatbot):
    """
    Reset 按钮点击处理：重置prompt

    参数：
        chatbot（List）：聊天记录
    返回：
        Tuple：更新后的聊天记录和清空的提示词文本
    """

    gr.Info("已清除提示词")
    chatbot.extend([["清除提示词","提示词已成功重置"]])
    return chatbot, "", 0

def assign_prompt(chatbot, prompt,template,example_number):
    """
    Assign 按钮点击处理：将用户输入的提示词分配给模型

    参数：
        chatbot（List）：聊天记录
        prompt（str）：用户输入的提示词
        template（str）: 当前的模板对象
        example_number（int）：选择的示例编号
    返回：
        Tuple：更新后的聊天记录、提示词文本、模板对象和选择的示例编号
    """

    gr.Info("正在分配提示词")
    example_number = int(example_number)
    token_num = my_model.prompt_token_num(prompt)#计算prompt的token数量
    if token_num > 1024:
        template = None
        gr.Warning("提示词过长，超过了1024个token")
        chatbot.append([None, "提示词太长，超过了1024个token"])
    elif example_number < 1 or example_number > len(questions):
        template = None
        prompt_ex = f"错误：请选择一个1到{len(questions)}之间的数字"
        gr.Warning(prompt_ex)
        chatbot.extend([[None, prompt_ex]])
    elif "{{question}}" not in prompt:
        template = None
        prompt_ex = "错误：提示词中必须包含占位符{{question}}"
        gr.Warning(prompt_ex)
        chatbot.extend([[None, prompt_ex]])
    else:
        environment = jinja2.Environment()
        template = environment.from_string(prompt)
        prompt_ex = f"""{template.render(questions=questions[example_number - 1])}"""
        chatbot.extend([["分配提示词", "提示词已成功分配\n\n自定义提示词示例："],[None, prompt_ex]])

    return chatbot, prompt, template, example_number, token_num


def clean_commas(text):
    """
    c处理数字中的逗号（千位分隔符）
    
    参数：
        text（str）：包含数字的文本
    返回：
        str：处理后的文本
    """
    def process_match(match):
        number = match.group(0)
        if '.' in number:
            return number
        else:
            number_list = number.split(",")
            new_string = number_list[0]
            for i in range(1,len(number_list)):
                if len(number_list[i]) == 3:
                    new_string += number_list[i]
                else:
                    new_string += f",{number_list[i]}"
            return new_string
    
    pattern = r'\d+(?:.\d+)*(?:\.\d+)'
    return re.sub(pattern,process_match, text)

def find_and_match(input_string,ground_truth):
    """
    检查输入中的数字是否与预期匹配
    
    参数：
        input_string（str）：包含数字的输入字符串
        ground_truth（str）：预期的正确数值
        
    返回：
        bool：如果找到匹配的数值则返回True，否则返回False
    """
    pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    found_numbers = pattern.findall(input_string)
    found_floats = [float(num) for num in found_numbers]
    return ground_truth in found_floats


def assess_prompt(chatbot, template, test_num):
    """
    Test按钮点击处理： 评估自定义prompt
    
    参数：
        chatbot（List）：聊天记录
        template（str）：当前的模板对象
        test_num（int）：要测试的问题数量
    
    返回：
        Tuple：更新后的聊天记录、结果列表、结果统计和UI组件
    """
    if template is None:
        chatbot.extend([[None, "评估失败，因此提示词模板为空（即无效的提示词）"]])
        gr.Warning("提示词未设置")
        return chatbot, [], "提示词未设置",gr.Slider(label="Result Number", value=0 ,minimum=0, maximum=0, step=1), gr.Textbox(label="Result",value="" , interactive=False)
    
    gr.Info("正在评估提示词")
    ans_template = "提示词和问题：\n\n{{question}}\n\n--------------------\n\n解题过程：\n\n{{rationale}}\n\n--------------------\n\n最终答案\n\n{{answer}}"
    res_list = []
    total_count = test_num
    environment = jinja2.Environment()
    ans_template = environment.from_string(ans_template)
    trial_num = 3
    trials = [[] for _ in range(trial_num)]
    res_stats_str = ""

    for i in range(trial_num):
        gr.Info(f"开始第{i+1}次测试")
        accurate_count = 0
        for idx, example in enumerate(questions[:test_num]):
            test_res = ""
            result = my_model.two_stage_completion(example, template.render(questions=example))

            if not result["answer"]:
                trials[i].append(0)
                test_res += f"第{i+1}次测试\n\n跳过问题{idx + 1}。"
                test_res += "\n" + "<"*6 + "="*30 + ">"*6 + "\n\n"
                res_list.append(f"第{i+1}次测试\n\n跳过问题{idx + 1}。")
                continue
        
            cleaned_result = clean_commas(result["answer"])
            if find_and_match(cleaned_result, answers[idx]):
                accurate_count += 1
                trials[i].append(1)
            else:
                trials[i].append(0)            

            my_model.save_cache()

            test_res += f"第{i+1}次测试\n\n"
            test_res += f"问题{idx + 1}：\n" + '-'*20
            test_res += f'''\n\n {ans_template.render(
                question=result['prompt'],
                rationale=result["rationale"],
                answer=result["answer"]
            )}\n'''
            test_res += "\n" + "<"*6 + "="*30 + ">"*6 + "\n\n"
            res_list.append(test_res)

        res_stats_str += f"第{i+1}次测试：正确数：{accurate_count},总数：{total_count}, 准确率：{accurate_count/total_count * 100}%\n"
        my_model.save_cache()

    voting_acc = 0
    for i in range(total_count):
        count = 0
        for j in range(trial_num):
            if trials[j][i] == 1:
                count += 1
        if count >= 2: 
            voting_acc += 1
    
    res_stats_str += f"最终准确率：{voting_acc / total_count * 100}%"
    chatbot.extend([["测试", "测试完成。结果可以在“结果”和“结果统计”中找到。"]])
    chatbot.extend([[None, "测试结果"], [None, ''.join(res_list)], [None, "结果统计"], [None, res_stats_str]])

    return chatbot, res_list, res_stats_str, gr.Slider(label="Result Number", value=1, minimum=1, maximum=len(res_list), step=1, visible=False), gr.Textbox(label="Result", value=res_list[0], interactive=False)


def save_prompt(chatbot, prompt):
    """
    Save 按钮点击处理：保存提示词
    
    参数：
        chatbot (List): 聊天记录
        prompt (str): 用户输入的提示词
    
    返回：
        List: 更新后的聊天记录
    """
    gr.Info("正在保存提示词")
    prompt_dict = {
        "prompt": prompt
    }

    with open("prompt.json", "w") as f:
        json.dump(prompt_dict, f)
    chatbot.extend([["保存提示词",f"提示词已保存为prompt.json"]])
    return chatbot

# Gradio UI
# Gradio界面
with gr.Blocks() as demo:
    my_magic_prompt = "任务：\n解决以下数学问题。\n\n问题：{{question}}\n\n答案："
    my_magic_prompt = my_magic_prompt.strip('\n')
    template = gr.State(None)
    res_list = gr.State(list())

    # 组件
    with gr.Tab(label="Console"):
        with gr.Group():
            example_num_box = gr.Dropdown(
                label="Demo Example (Please choose one example for demo)",
                value=1,
                info=questions[0],
                choices=[i+1 for i in range(len(questions))],
                filterable=False
            )
            prompt_textbox = gr.Textbox(
                label="Custom Prompt",
                placeholder=f"在这里输入你的自定义提示词。例如：\n\n{my_magic_prompt}",
                value="",
                info="请确保包含`{{question}}`标签。"
            )
            with gr.Row():
                set_button = gr.Button(value="Set Prompt")
                reset_button = gr.Button(value="Clear Prompt")
            prompt_token_num = gr.Textbox(
                label="Number of prompt tokens",
                value=0,
                interactive=False,
                info="自定义提示词的Token数量。"
            )
        with gr.Group():
            test_num = gr.Slider(
                label="Number of examples used for evaluation",
                minimum=1,
                maximum=len(questions),
                step=1,
                value=1
            )
            assess_button = gr.Button(value="Evaluate")
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        trial_no = gr.Slider(
                            label="Trial ID",
                            value=1,
                            minimum=1,
                            maximum=3,
                            step=1
                        )
                        ques_no = gr.Slider(
                            label="Question ID",
                            value=1,
                            minimum=1,
                            maximum=1,
                            step=1
                        )
                    res_num = gr.Slider(
                        label="Result Number",
                        value=0,
                        minimum=0,
                        maximum=0,
                        step=1,
                        visible=False
                    )
                    res = gr.Textbox(
                        label="Result",
                        value="",
                        placeholder="暂无结果",
                        interactive=False
                    )
                with gr.Column():
                    res_stats = gr.Textbox(label="Result Stats", interactive=False)
            save_button = gr.Button(value="Save Custom Prompt")
    with gr.Tab(label="Log"):
        chatbot = gr.Chatbot(label="Log")

    # 事件处理
    example_num_box.input(
        lambda example_number: gr.Dropdown(
            label="Example (Please choose one example for demo)",
            value=example_number,
            info=questions[example_number - 1],
            choices=[i+1 for i in range(len(questions))]
        ),
        inputs=[example_num_box],
        outputs=[example_num_box]
    )
    
    res_num.change(
        lambda results, result_num, test_num: (
            gr.Textbox(label="Result", value=results[result_num-1], interactive=False) 
            if len(results) != 0 
            else gr.Textbox(label="Result", value="", placeholder="暂无结果", interactive=False),
            (int)((result_num-1)/test_num)+1,
            gr.Slider(
                label="Question Number", 
                minimum=1, 
                maximum=test_num, 
                value=(result_num-1)%test_num+1, 
                step=1
            )
        ),
        inputs=[res_list, res_num, test_num],
        outputs=[res, trial_no, ques_no]
    )
    
    trial_ques_no_input = lambda t_val, q_val, test_num: (t_val - 1) * test_num + (q_val if q_val is not None else 0)
    trial_no.input(trial_ques_no_input, inputs=[trial_no, ques_no, test_num], outputs=[res_num])
    ques_no.input(trial_ques_no_input, inputs=[trial_no, ques_no, test_num], outputs=[res_num])
    set_button.click(assign_prompt, inputs=[chatbot, prompt_textbox, template, example_num_box], outputs=[chatbot, prompt_textbox, template, example_num_box, prompt_token_num])
    reset_button.click(reset_prompt, inputs=[chatbot], outputs=[chatbot, prompt_textbox, prompt_token_num])
    assess_button.click(assess_prompt, inputs=[chatbot, template, test_num], outputs=[chatbot, res_list, res_stats, res_num, res])
    save_button.click(save_prompt, inputs=[chatbot, prompt_textbox], outputs=[chatbot])

demo.queue().launch(debug=True)