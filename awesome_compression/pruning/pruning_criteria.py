import copy
import math
import random
import time

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

# 设置 matplotlib 使用支持负号的字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 定义一个LeNet网络
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device=device)

# 加载模型的状态字典
checkpoint = torch.load('model.pt')
# 加载状态字典到模型
model.load_state_dict(checkpoint)
origin_model = copy.deepcopy(model)


# 绘制权重分布图
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    # 删除多余的子图
    fig.delaxes(axes[1][2])

    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        # 只选择维度大于1的参数（即权重矩阵）
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color='green', alpha=0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color='green', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.800)
    plt.show()

plot_weight_distribution(model)


# 计算每一层网络的稠密程度
def plot_num_parameters_distribution(model):
    num_parameters = dict()
    num_nonzeros, num_elements = 0, 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_nonzeros = param.count_nonzero()
            num_elements = param.numel()  # numel() number of elements
            dense = float(num_nonzeros) / num_elements
            num_parameters[name] = dense
    # 创建一个图形窗口（通常这一步是隐式的，但你可以显式地调用它）
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y') # 在y轴上添加网格线，以便更清晰地查看数据。

    bars = plt.bar(list(num_parameters.keys()), list(num_parameters.values()))

    # 在柱状图上添加数据标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, yval, va='bottom')  # va='bottom' 使得文本在柱状图上方

    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60) # 设置x轴刻度标签的旋转角度为60度，以便在参数名称较长时能够清晰显示。
    plt.tight_layout()
    plt.show()
# 列出weight直方图
plot_num_parameters_distribution(model)

""" base on L1 """
@torch.no_grad()
def prune_l1(weight, percentile=0.5):
    num_elements = weight.numel()

    # 计算值为0的数量
    num_zeros = round(num_elements * percentile)
    # 计算weight的重要性
    importance = weight.abs()
    # 计算裁剪阈值
    threshold = importance.view(-1).kthvalue(num_zeros).values
    # 计算mask
    mask = torch.gt(importance, threshold)

    # 计算mask后的weight
    weight.mul_(mask)
    return weight

# 裁剪conv2层
weight_pruned = prune_l1(model.conv2.weight, percentile=0.5)

# 替换原有model层
model.conv2.weight.data = weight_pruned

# 列出weight直方图
plot_weight_distribution(model)
plot_num_parameters_distribution(model)

""" base on L2 """
@torch.no_grad()
def prune_l2(weight, percentile=0.5):
    num_elements = weight.numel()

    # 计算值为0的数量
    num_zeros = round(num_elements * percentile)
    # 计算weight的重要性（使用L2范数，即各元素的平方）
    importance = weight.pow(2)
    # 计算裁剪阈值
    threshold = importance.view(-1).kthvalue(num_zeros).values
    # 计算mask
    mask = torch.gt(importance, threshold)

    # 计算mask后的weight
    weight.mul_(mask)
    return weight

# 裁剪fc1层
weight_pruned = prune_l2(model.fc1.weight, percentile=0.4)
# 替换原有model层
model.fc1.weight.data = weight_pruned
# 列出weight直方图
plot_weight_distribution(model)
# 列出weight直方图
plot_num_parameters_distribution(model)
# 保存裁剪后的weight
torch.save(model.state_dict(), './model_pruned.pt')