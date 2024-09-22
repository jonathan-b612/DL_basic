import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))  #x.shape(400,2)
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色 #y.shape(400,1)
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

seq_net = Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4,1)
)

#get the w0's weight
# w0 = seq_net[0].weight
# print(w0)

param = seq_net.parameters()
optim = torch.optim.SGD(param,lr = 1.)
criterion = nn.BCEWithLogitsLoss()
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# 我们训练 10000 次
for e in range(10000):
    out = seq_net(x)
    loss = criterion(out, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.item()))


def plot_decision_boundary(model, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them #shape(1008,1030)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()]) #input_shape(1038240,2) #output_shape(1038240,1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)


def plot_network(x):
    x = torch.from_numpy(x).float()
    x0 = seq_net(x)
    out = F.sigmoid(x0)
    out = (out > 0.5) * 1
    return out.data.numpy()

print(type(x))
plot_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
plt.title('sequential network')
plt.show()


"""
 save the parameter&&model
# 将参数和模型保存在一起
torch.save(seq_net, 'save_seq_net.pth')
# 读取保存的模型
seq_net1 = torch.load('save_seq_net.pth')
print(seq_net1)
print(seq_net1[0].weight)
"""

# generally, we use the method of "state_dict()" to save parameters
# 保存模型参数
print(seq_net.state_dict())
torch.save(seq_net.state_dict(), 'save_seq_net_params.pth')
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

seq_net2.load_state_dict(torch.load('save_seq_net_params.pth'))