"""
Pytorch在1.4.0版本开始，加入了剪枝操作，在torch.nn.utils.prune模块中，本教程按照剪枝范围划分，将其分以下几种剪枝方式:

局部剪枝（Local Pruning）
结构化剪枝
随机结构化剪枝（random_structured）
范数结构化剪枝（ln_structured）
非结构化剪枝
随机非结构化剪枝（random_unstructured）
范数非结构化剪枝（l1_unstructured）
全局剪枝（Global Pruning）
非结构化剪枝（global_unstructured）
自定义剪枝（Custom Pruning）
注： 全局剪枝只有非结构化剪枝方式。
"""

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchsummary import summary

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

# 打印模型结构
summary(model, input_size=(1, 28, 28))

# 打印第一个卷积层的参数
module = model.conv1
print(list(module.named_parameters()))

# 打印module中的属性张量named_buffers，初始时为空列表
print(list(module.named_buffers()))

# 打印模型的状态字典，状态字典里包含了所有的参数
print(model.state_dict().keys())

# 第一个参数: module, 代表要进行剪枝的特定模块, 这里指的是module=model.conv1,
#             说明这里要对第一个卷积层执行剪枝.
# 第二个参数: name, 代表要对选中的模块中的哪些参数执行剪枝.
#             这里设定为name="weight", 说明是对网络中的weight剪枝, 而不对bias剪枝.
# 第三个参数: amount, 代表要对模型中特定比例或绝对数量的参数执行剪枝.
#             amount是一个介于0.0-1.0的float数值,代表比例, 或者一个正整数，代表指定剪裁掉多少个参数.
# 第四个参数: dim, 代表要进行剪枝通道(channel)的维度索引.
"""
 import torch.nn.utils.prune as prune 
结论: 经过剪枝操作后, 原始的权重矩阵weight变成了weight_orig. 
并且剪枝前打印为空列表的module.named_buffers(), 现在多了weight_mask参数.
"""
prune.random_structured(module, name="weight", amount=2, dim=0)

# 再次打印模型的状态字典，观察conv1层
print(model.state_dict().keys())

# 再次打印模型的参数named_parameters
print(list(module.named_parameters()))

# 再次打印module中的属性张量named_buffers
print(list(module.named_buffers()))

# 打印_forward_pre_hooks
print(module._forward_pre_hooks)
