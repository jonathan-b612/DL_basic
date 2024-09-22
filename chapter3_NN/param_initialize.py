import numpy as np
import torch
from torch import nn

# 定义一个 Sequential 模型
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# 访问第一层的参数
w1 = net1[0].weight
b1 = net1[0].bias

print(w1)

# 定义一个 Tensor 直接对其进行替换
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(40, 30)))

print(w1)