import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('../data/小丸子.png').convert("L")# 读入一张灰度图的图片
im = np.array(im, dtype='float32') # 将其转换为一个矩阵

im = torch.from_numpy(im.reshape(1,1,im.shape[0],im.shape[1]))

# # nn.Conv2d
# conv1 = nn.Conv2d(1, 1, 3, bias=False) # 定义卷积
#
# sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') # 定义轮廓检测算子
# sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3)) # 适配卷积的输入输出
# conv1.weight.data = torch.from_numpy(sobel_kernel) # 给卷积的 kernel 赋值
# edge1 = conv1(im) # 作用在图片上
# edge1 = edge1.detach().squeeze().numpy() # 将输出转换为图片的格式
# plt.imshow(edge1, cmap='gray')
# plt.show()

# #nn.functional.Conv2d
# sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') # 定义轮廓检测算子
# sobel_kernel = torch.from_numpy(sobel_kernel.reshape((1, 1, 3, 3))) # 适配卷积的输入输出
# edge1 = nn.functional.conv2d(im,sobel_kernel)
# edge1 = edge1.detach().squeeze().numpy()
# plt.imshow(edge1, cmap='gray')
# plt.show()

##nn.MaxPool2d
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
print('before max pool,image shape: {}x {}'.format(im.shape[2], im.shape[3]) )
small_im = pool1(im)
small_im = small_im.detach().squeeze().numpy()
print('after max pool,image shape: {} x {}'.format(small_im.shape[0], small_im.shape[1]) )
plt.imshow(small_im,cmap='gray')
plt.show()

