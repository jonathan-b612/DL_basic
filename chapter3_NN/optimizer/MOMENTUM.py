import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def sgd_momentum(parameters, vs, lr, gamma):
    for param, v in zip(parameters, vs):
        v[:]= gamma * v - lr * param.grad.data
        param.data = param.data + v


def data_tf(x):
    x = np.array(x, dtype='float32') / 255 # 将数据变到 0 ~ 1 之间
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

train_dataset = MNIST('../../data', train=True, transform=data_tf, download=True) # 载入数据集，申明定义的数据变换
test_dataset = MNIST('../../data', train=False, transform=data_tf, download=True)

dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# 定义 loss 函数
criterion = nn.CrossEntropyLoss()


data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 使用 Sequential 定义 3 层神经网络
net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)
#equal to sgd_moment
#optmizer = torch.optim.SGD(net.parameters(), lr=1e-2,momentum=0.9) # 加动量i

# 将速度初始化为和参数形状相同的零张量
vs = []
for param in net.parameters():
    vs.append(torch.zeros_like(param.data))

# 开始训练
losses = []
idx = 0
i = 0
"""
具体来说，
nn.CrossEntropyLoss 会自动对 out 应用 softmax 函数
（但不是显式地，而是在计算损失时内部处理），
然后对每个样本计算其对应类别的负对数似然损失。
这里的 label 用于指定每个样本的真实类别，
以便损失函数可以正确地计算损失。
print(out.shape)
print(label.shape)
out：
torch.Size([64, 10])
torch.Size([64])
"""
start = time.time()  # 记时开始
for e in range(5):
    train_loss = 0
    for im, label in dataloader:
        # 前向传播
        out = net(im)

        loss = criterion(out, label)
        # 反向传播
        net.zero_grad()
        loss.backward()
        sgd_momentum(net.parameters(), vs, 1e-2, 0.9)  # 使用的动量参数为 0.9，学习率 0.01
        #optimizer.step()
        # 记录误差
        train_loss += loss.item()
        if idx % 30 == 0:
            losses.append(loss.item())
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_dataset)))
end = time.time()  # 计时结束
print('使用时间: {:.5f} s'.format(end - start))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#使用 Sequential 定义 3 层神经网络
net1 = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)

optimizer = torch.optim.SGD(net1.parameters(), lr=1e-2) # 不加动量

# 开始训练
idx = 0
losses1 = []
start = time.time() # 记时开始
for e in range(5):
    train_loss = 0
    for im, label in dataloader:
        # 前向传播
        out = net1(im)
        loss1 = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss1.item()
        if idx % 30 == 0: # 30 步记录一次
            losses1.append(loss1.item())
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_dataset)))
end = time.time() # 计时结束
print('使用时间: {:.5f} s'.format(end - start))

x_axis = np.linspace(0, 5, len(losses), endpoint=True)
plt.semilogy(x_axis, losses, label='momentum: 0.9')
plt.semilogy(x_axis, losses1, label='no momentum')
plt.legend(loc='best')
plt.show()