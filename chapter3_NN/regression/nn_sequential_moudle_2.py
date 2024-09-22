import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D)) #x.shape(400,2)
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色 #y.shape(400,1)
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

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

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

w1 = nn.Parameter(torch.randn(2,4)*0.01)
b1 = nn.Parameter(torch.zeros(4))

w2 = nn.Parameter(torch.randn(4,1)*0.01)
b2 = nn.Parameter(torch.zeros(1))

def two_network(x):
    x1 = torch.mm(x, w1)+b1
    x1 =F.tanh(x1)
    x2 = torch.mm(x1, w2)+b2
    return x2

optimizer = torch.optim.SGD([w1, w2,b1,b2], lr=1.)
criterion = nn.BCEWithLogitsLoss()

for e in range(10000):
    out = two_network(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e+1) % 1000 == 0:
        print('Epoch: {}, Loss: {}'.format(e+1, loss.item()))

def plot_network(x):
    x = torch.from_numpy(x).float()
    x1 = torch.mm(x, w1) + b1
    x1 = F.tanh(x1)
    x2 = torch.mm(x1, w2) + b2
    out = F.sigmoid(x2)
    out = (out > 0.5) * 1
    return out.data.numpy()

plot_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
plt.title('2 layer network')
plt.show()