# 以 AutoModelForQuestionAnswering 为例，使用 inspect 库查看对应源码：
# 查看__init__方法

import inspect
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# 获取并打印__init__方法的源码
init_code = inspect.getsource(model.__init__)
# print(init_code)

# 查看forward方法
forward_code = inspect.getsource(model.forward)
# print(forward_code)

help(AutoModelForQuestionAnswering)

