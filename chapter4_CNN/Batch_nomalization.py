import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist


def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 数据预处理，标准化
    x = x.reshape(-1,) # 拉平
    x = torch.from_numpy(x)
    return x



train_dataset = mnist.MNIST('../data', train=True, transform=data_tf, download=True)  # 重新载入数据集，申明定义的数据变换
test_dataset = mnist.MNIST('../data', train=False, transform=data_tf, download=True)


train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=128, shuffle=False)


class multi_network(nn.Module):
    def  __init__(self):
        super(multi_network, self).__init__()
        self.layer1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100, 10)

        self.gamma = nn.Parameter(torch.randn(100))
        self.beta = nn.Parameter(torch.randn(100))

        self.moving_mean = torch.zeros(100)
        self.moving_var = torch.zeros(100)

    def forward(self, x,is_train=True):
            x = self.layer1(x)
            x = batch_norm_1d(x, gamma=self.gamma, beta=self.beta, is_training=is_train, moving_mean=self.moving_mean, moving_var=self.moving_var)
            x = self.relu(x)
            x = self.layer2(x)
            return x


net = multi_network()
# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 0.1) # 使用随机梯度下降，学习率 0.1

from utils import train
train(net,train_data,test_data,criterion,optimizer,10)


# 打出 moving_mean 的前 10 项
print(net.moving_mean[:10])

print(""""2.no batch normalization""")
no_bn_net = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

optimizer = torch.optim.SGD(no_bn_net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1
train(no_bn_net, train_data,test_data,criterion,optimizer,10)


print("""3.use batch normalization in convolutional network""")
def data_tf2(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 数据预处理，标准化
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    return x

train_dataset2 = mnist.MNIST('../data', train=True, transform=data_tf2, download=True) # 重新载入数据集，申明定义的数据变换
test_dataset2 = mnist.MNIST('../data', train=False, transform=data_tf2, download=True)
train_data2 = DataLoader(train_dataset2, batch_size=64, shuffle=True)
test_data2 = DataLoader(test_dataset2, batch_size=128, shuffle=False)

# 使用批标准化
class conv_bn_net(nn.Module):
    def __init__(self):
        super(conv_bn_net, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classfy = nn.Linear(400, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        x = self.classfy(x)
        return x


conv_bn_net = conv_bn_net()
optimizer2 = torch.optim.SGD(conv_bn_net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1

train(conv_bn_net,train_data2,test_data2,criterion,optimizer2,10)

print("""4.no use batch normalization in convolutional network""")


# 不使用批标准化
class conv_no_bn_net(nn.Module):
    def __init__(self):
        super(conv_no_bn_net, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classfy = nn.Linear(400, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        x = self.classfy(x)
        return x


conv_no_bn_net = conv_no_bn_net()
optimizer3 = torch.optim.SGD(conv_no_bn_net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
train(conv_no_bn_net,train_data2,test_data2,criterion,optimizer3,10)