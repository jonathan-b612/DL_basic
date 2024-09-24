import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU()]  # 定义第一层

    for i in range(num_convs - 1):  # 定义后面的很多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU())

    net.append(nn.MaxPool2d(2, 2))  # 定义池化层
    return nn.Sequential(*net)

block_demo = vgg_block(3, 64, 128)
#print(block_demo)

def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
        #*net
    return nn.Sequential(*net)

vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
print(vgg_net)

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


from utils import train


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.transpose((2, 0, 1))  # 将 channel 放到第一维，只是 pytorch 要求的输入方式
    x = torch.from_numpy(x)
    return x


train_set = CIFAR10('../data', train=True, transform=data_tf,download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('../data', train=False, transform=data_tf,download=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

net = vgg()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()

from utils import train
train(net,train_data,test_data,criterion,optimizer,5)