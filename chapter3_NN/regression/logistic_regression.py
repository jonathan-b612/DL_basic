import numpy as np
import torch
from matplotlib import pyplot as plt
from prometheus_client.decorator import init
from torch import nn

""""
为什么要将数据标准化？
为什么将学习速率从0.0001调整至1？
"""

with open(r'D:\python_project\jupyter_py\code-of-learn-deep-learning-with-pytorch\chapter3_NN\logistic-regression\data.txt','r') as f:

    """
    类型：list
    内容：每个元素都是文件 f 中的一行，作为字符串（str）存储在列表中。
    如果文件 f 包含多行，那么 data_list 将是一个包含多个字符串的列表，每个字符串对应文件中的一行。
    如果文件是空的，那么 data_list 将是一个空列表。
    """
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

# 标准化
x0_max = max([i[0] for i in data])

x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]


# x0 = list(filter(lambda x : x[-1] == 0, data))
# x1 = list(filter(lambda x : x[-1] == 1, data))
"""
filter 函数的两个主要参数分别是：
函数：这是一个函数对象，
它接受一个参数（通常是我们想要从可迭代对象中过滤的元素），
并返回一个布尔值（True 或 False）。
这个函数的返回值决定了元素是否应该被包含在过滤后的结果中。
具体来说，如果函数返回 True，则当前元素会被包含在结果中；
如果返回 False，则当前元素会被排除。

可迭代对象：这是一个序列（如列表、元组、字符串等）或其他可迭代对象，
它包含了要被过滤的原始元素。filter 函数会遍历这个可迭代对象中的每个元素，
并将它们作为参数传递给前面提到的函数，以决定哪些元素应该被保留。
"""

# plot_x0_0 = [i[0] for i in x0]
# plot_x0_1 = [i[1] for i in x0]
# plot_x1_0 = [i[0] for i in x1]
# plot_x1_1 = [i[1] for i in x1]
#
# plt.scatter(plot_x0_0, plot_x0_1, color='blue',label='x0')
# plt.scatter(plot_x1_0, plot_x1_1, color='red',label='x1')
# plt.legend(loc='best')
# plt.show()

torch.manual_seed(2024)
#for the train
np_data = np.array(data, dtype='float32') # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2])# 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) # 转换成 Tensor，大小是 [100, 1]


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


logistic_regression_model = LogisticRegression()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_regression_model.parameters(), lr=1)


import time
start_time = time.time()

for epoch in range(10000):
    y_pred = logistic_regression_model(x_data)
    loss = criterion(y_pred, y_data)
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().item() / y_data.shape[0]
    if epoch % 200 == 0:
        print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch+1, loss.item(), acc))
        weight = []
        b = []
        for name, param in logistic_regression_model.named_parameters():
            if 'weight' in name:
                # 打印权重，注意weight是一个包含所有权重的张量，对于nn.Linear(2,1)
                print(f"Weight (w1, w2): {param.cpu().detach().numpy().flatten()}")
                weight.append(param.cpu().detach().numpy().flatten())
            elif 'bias' in name:
                # 打印偏置项
                print(f"Bias (b): {param.cpu().detach().numpy()[0]}")
                b.append(param.cpu().detach().numpy()[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

during = time.time() - start_time
print()
print('During Time: {:.3f} s'.format(during))

