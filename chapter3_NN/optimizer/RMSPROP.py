import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def rmsprop(parameters, sqrs, lr, alpha):
    eps = 1e-10
    for param, sqr in zip(parameters, sqrs):
        sqr[:] = alpha * sqr + (1 - alpha) * param.grad.data ** 2
        div = lr / torch.sqrt(sqr + eps) * param.grad.data
        print(torch.sqrt(sqr + eps).shape, div.shape)
        param.data = param.data - div

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

sqrs = []
for param in net.parameters():
    sqrs.append(torch.zeros_like(param.data))

# 开始训练
losses = []
idx = 0
start = time.time() # 记时开始
for e in range(5):
    train_loss = 0
    for im, label in dataloader:
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        net.zero_grad()
        loss.backward()
        #optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.5)
        #optimizer.step()
        rmsprop(parameters=net.parameters(), sqrs=sqrs, lr=0.0001, alpha=0.5)
        # 记录误差
        train_loss += loss.item()
        if idx % 30 == 0:
            losses.append(loss.item())
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_dataset)))
end = time.time() # 计时结束
print('使用时间: {:.5f} s'.format(end - start))

x_axis = np.linspace(0, 5, len(losses), endpoint=True)
plt.semilogy(x_axis, losses, label='adagrad-RMSPROP-alpha=0.5')
plt.legend(loc='best')
plt.show()

