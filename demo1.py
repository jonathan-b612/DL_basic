'''
import torch

#create variable
x = torch.tensor([1], dtype=torch.float,requires_grad=True)
w = torch.tensor([2], dtype=torch.float,requires_grad=True)
b = torch.tensor([3], dtype=torch.float,requires_grad=True)

# build a comutational graph
y = w * x + b

y.backward()
print(x.grad)
print(w.grad)
print(b.grad)
'''
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
""""#tensor.mean()"""
# import torch
#
#
# input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
#
#                              [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
#
#                              [[17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]], dtype=torch.float32)
#
#
# mean_along_dim0 = input_tensor.mean(dim=2,keepdim=True)
#
# print(mean_along_dim0)
# final_mean = mean_along_dim0.mean(dim=0)
# print(final_mean)
#
# mean_values_dim2 = input_tensor.mean(dim=[0, 1])
#
# print(mean_values_dim2)

#tensor.mean()-keepdim
# import torch
#
# # 创建一个形状为 (2, 1, 3) 的三维张量
#
# input_tensor = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]],dtype=torch.float)
#
# # 使用mean函数归约维度1，同时设置keepdim=True
#
# mean_with_keepdim = input_tensor.mean(dim=1, keepdim=True)
#
# # 使用mean函数归约维度1，但不设置keepdim（默认为False）
#
# mean_without_keepdim = input_tensor.mean(dim=1)
#
# # 打印结果
#
# print("With keepdim=True:")
#
# print(mean_with_keepdim.shape)  # 输出: torch.Size([2, 1, 3])
#
# print(mean_with_keepdim)
#
# print("\nWith keepdim=False (default):")
#
# print(mean_without_keepdim.shape)  # 输出: torch.Size([2, 3])
#
# print(mean_without_keepdim)

"""#view view_as"""
# import torch
#
# # 假设我们有两个张量x和y，它们的元素数量相同但形状不同
# x = torch.randn(2, 3)  # 形状为(2, 3)的张量
# y = torch.randn(3, 2)  # 形状为(3, 2)的张量，注意这里我们不能直接使用view_as，因为x和y的元素数量不同
#
# # 为了让x和某个张量形状相同，我们可以先调整x的形状
# x_reshaped = x.view(3, 2)  # 现在x_reshaped的形状是(3, 2)，与y相同
# print(x)
# print(x_reshaped)

# import numpy as np
#
# # 创建一个一维数组a
# a = np.array([1, 2, 3])
# # 创建一个二维数组b，其中第一维是1（长度为1的维度）
# b = np.array([[10, 20, 30]])
#
# # 使用广播进行元素级加法
# c = a + b
#
# print(c)
# # 输出:
# # [[11 22 33]

"""torch.argmax"""
# a = torch.randn(2,2)
# print(a)
# print(torch.argmax(a).shape)
#
# import torch
#
# # 假设的预测概率输出
# predictions = torch.tensor([
#     [0.9, 0.05, 0.05],  # 图像1
#     [0.01, 0.98, 0.01],  # 图像2
#     [0.02, 0.03, 0.95],  # 图像3
#     [0.8, 0.15, 0.05],  # 图像4
#     [0.05, 0.9, 0.05]  # 图像5
# ])
#
# # 沿着第一个维度（dim=0）进行argmax
# pred_label_wrong_dim = predictions.argmax(dim=0)
#
# # 打印结果
# print(pred_label_wrong_dim)  # 输出可能是类似于 tensor([4, 1, 3]) 的东西，但这取决于概率的具体值


transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像或 NumPy ndarray 转换为 torch.Tensor
])

def data_tf2(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 数据预处理，标准化
    x = torch.from_numpy(x)
    print(x.shape)
    x = x.unsqueeze(0)
    return x

train_dataset2 = mnist.MNIST('./data', train=True, transform=data_tf2,download=True) # 重新载入数据集，申明定义的数据变换
test_dataset2 = mnist.MNIST('./data', train=False,transform=data_tf2,download=True)
train_data2 = DataLoader(train_dataset2, batch_size=64, shuffle=True)
test_data2 = DataLoader(test_dataset2, batch_size=128, shuffle=False)
for i,j in train_data2:
    print(i.shape,j.shape)