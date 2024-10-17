import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# 创建一个可视化2维矩阵函数，将值为0的元素与其他区分开
""" draw picture """
def plot_tensor(tensor, title):
    # 创建一个新的图像和轴
    fig, ax = plt.subplots()

    # 使用 CPU 上的数据，转换为 numpy 数组，并检查相等条件，设置颜色映射
    ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # 遍历矩阵中的每个元素并添加文本标签
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            text = ax.text(j, i, f'{tensor[i, j].item():.2f}', ha="center", va="center", color="k")

    # 显示图像
    plt.show()

# 创建一个矩阵weight
weight = torch.rand(8, 8)
plot_tensor(weight, '剪枝前weight')

# 细粒度剪枝方法1
""" method_0 """
def _fine_grained_prune(tensor: torch.Tensor, threshold  : float) -> torch.Tensor:
    """
    :param tensor: 输入张量，包含需要剪枝的权重。
    :param threshold: 阈值，用于判断权重的大小。
    :return: 剪枝后的张量。
    """
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            if tensor[i, j] < threshold:
                tensor[i][j] = 0
    return tensor

#pruned_weight = _fine_grained_prune(weight, 0.5)
#plot_tensor(weight, '细粒度剪枝后weight')

# 细粒度剪枝方法2
''' method_1 '''
def fine_grained_prune(tensor: torch.Tensor, threshold  : float) -> torch.Tensor:
    """
    创建一个掩码张量，指示哪些权重不应被剪枝（应保持非零）。

    :param tensor: 输入张量，包含需要剪枝的权重。
    :param threshold: 阈值，用于判断权重的大小。
    :return: 剪枝后的张量。
    """
    mask = torch.gt(tensor, threshold)
    tensor.mul_(mask)
    return tensor
#pruned_weight = fine_grained_prune(weight, 0.5)
#plot_tensor(pruned_weight, '细粒度剪枝后weight')

from itertools import permutations

# 细粒度剪枝方法3-基于模式的剪枝
""" base on pattern"""
def reshape_1d(tensor, m):
    # 转换成列为m的格式，若不能整除m则填充0
    if tensor.shape[1] % m > 0:
        mat = torch.FloatTensor(tensor.shape[0], tensor.shape[1] + (m - tensor.shape[1] % m)).fill_(0)
        mat[:, : tensor.shape[1]] = tensor
        return mat.view(-1, m)
    else:
        return tensor.view(-1, m)

def compute_valid_1d_patterns(m, n):
    patterns = torch.zeros(m)
    patterns[:n] = 1
    # permutations A(5up,5down) = 5*4*3*2*1
    valid_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
    return valid_patterns

def compute_mask(tensor, m, n):
    # 计算所有可能的模式
    patterns = compute_valid_1d_patterns(m,n)
    #print(patterns)

    # 找到m:n最好的模式
    mask = torch.IntTensor(tensor.shape).fill_(1).view(-1,m)
    mat = reshape_1d(tensor, m)
    pmax = torch.argmax(torch.matmul(mat.abs(), patterns.t()), dim=1)
    #print(pmax)
    mask[:] = patterns[pmax[:]]
    #print(mask)
    mask = mask.view(tensor.shape)
    return mask

def pattern_pruning(tensor, m, n):
    mask = compute_mask(weight, m, n)
    tensor.mul_(mask)
    return tensor

pruned_weight = pattern_pruning(weight, 4, 2)
plot_tensor(pruned_weight, '剪枝后weight')

# 细粒度剪枝方法4-向量级剪枝
""" base on vector """
# 剪枝某个点所在的行与列
def vector_pruning(weight, point):
    row, col = point
    prune_weight = weight.clone()
    prune_weight[row, :] = 0
    prune_weight[:, col] = 0
    return prune_weight
point = (1, 1)
#prune_weight = vector_pruning(weight, point)
#plot_tensor(prune_weight, '向量级剪枝后weight')

#细粒度剪枝方法5-卷积核级剪枝
""" base on convolution kernel"""
""" (outchannel, inchannel, width, height) """
# 定义可视化4维张量的函数
def visualize_tensor(tensor, title, batch_spacing=3):
    fig = plt.figure()  # 创建一个新的matplotlib图形
    ax = fig.add_subplot(111, projection='3d')  # 向图形中添加一个3D子图

    # 遍历张量的批次维度
    for batch in range(tensor.shape[0]):
        # 遍历张量的通道维度
        for channel in range(tensor.shape[1]):
            # 遍历张量的高度维度
            for i in range(tensor.shape[2]):
                # 遍历张量的宽度维度
                for j in range(tensor.shape[3]):
                    # 计算条形的x位置，考虑到不同批次间的间隔
                    x = j + (batch * (tensor.shape[3] + batch_spacing))
                    y = i  # 条形的y位置，即张量的高度维度
                    z = channel  # 条形的z位置，即张量的通道维度
                    # 如果张量在当前位置的值为0，则设置条形颜色为红色，否则为绿色
                    color = 'red' if tensor[batch, channel, i, j] == 0 else 'green'
                    # 绘制单个3D条形
                    ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=color, edgecolor='black', alpha=0.9)

    ax.set_title(title)  # 设置3D图形的标题
    ax.set_xlabel('Width')  # 设置x轴标签，对应张量的宽度维度
    ax.set_ylabel('Height')  # 设置y轴标签，对应张量的高度维度
    ax.set_zlabel('Channel')  # 设置z轴标签，对于张量的通道维度
    ax.set_zlim(ax.get_zlim()[::-1])  # 反转z轴方向
    ax.zaxis.labelpad = 15  # 调整z轴标签的填充
    plt.show()  # 显示图形


def prune_conv_layer(conv_layer, prune_method, title="", percentile=0.2, vis=True):
    prune_layer = conv_layer.clone()

    l2_norm = None
    mask = None

    # 计算每个kernel的L2范数 (dim = (-2,-1) , (width,height))
    # l2_norm.shape (batch,channel,1,1)
    l2_norm = torch.norm(prune_layer, p=2, dim=(-2, -1), keepdim=True)
    # threshold.shape torch.size() - 0 vector
    threshold = torch.quantile(l2_norm, percentile)
    mask = l2_norm > threshold
    print(mask)
    print(prune_layer)
    prune_layer = prune_layer * mask.float()
    print(prune_layer)
    visualize_tensor(prune_layer, title=prune_method)


# 使用PyTorch创建一个张量
# tensor = torch.rand((3, 10, 4, 5))

# 调用函数进行剪枝
# pruned_tensor = prune_conv_layer(tensor, 'Kernel级别剪枝', vis=True)

# 细粒度剪枝方法6-滤波器级别剪枝
""" base on filter """
""" (outchannel, inchannel, width, height) """
def prune_conv_layer(conv_layer, prune_method, title="", percentile=0.2, vis=True):
    prune_layer = conv_layer.clone()

    l2_norm = None
    mask = None

    # 计算每个Filter的L2范数
    l2_norm = torch.norm(prune_layer, p=2, dim=(1, 2, 3), keepdim=True)
    threshold = torch.quantile(l2_norm, percentile)
    mask = l2_norm > threshold
    print(mask)
    print(mask.shape)
    prune_layer = prune_layer * mask.float()

    visualize_tensor(prune_layer, title=prune_method)


# 使用PyTorch创建一个张量
#tensor = torch.rand((3, 10, 4, 5))

# 调用函数进行剪枝

# pruned_tensor = prune_conv_layer(tensor, 'Filter级别剪枝', vis=True)

#细粒度剪枝方法7-通道数级别剪枝
""" base on channel"""
def prune_conv_layer(conv_layer, prune_method, title="", percentile=0.2, vis=True):
    prune_layer = conv_layer.clone()

    l2_norm = None
    mask = None

    # 计算每个channel的L2范数
    l2_norm = torch.norm(prune_layer, p=2, dim=(0, 2, 3), keepdim=True)
    threshold = torch.quantile(l2_norm, percentile)
    mask = l2_norm > threshold
    prune_layer = prune_layer * mask.float()

    visualize_tensor(prune_layer, title=prune_method)


# 使用PyTorch创建一个张量
# tensor = torch.rand((3, 10, 4, 5))

# 调用函数进行剪枝

# pruned_tensor = prune_conv_layer(tensor, 'Channel级别剪枝', vis=True)

# 返回一个mask
def get_threshold_and_mask(norms, percentile):
    threshold = torch.quantile(norms, percentile)
    return norms > threshold

def prune_conv_layer(conv_layer, prune_method, title= "", percentile=0.2, vis=True):
    prune_layer = conv_layer.clone()
    mask = None
    if prune_method == "fine_grained":
        prune_layer[torch.abs(prune_layer) < percentile] = 0
    elif prune_method == "vector_level":
        mask = get_threshold_and_mask(torch.norm(prune_layer, p=2, dim=-1), percentile).unsqueeze(-1)
    elif prune_method == "kernel_level":
        mask = get_threshold_and_mask(torch.norm(prune_layer, p=2, dim=(-2, -1), keepdim=True), percentile)
    elif prune_method == "filter_level":
        mask = get_threshold_and_mask(torch.norm(prune_layer, p=2, dim=(1, 2, 3), keepdim=True), percentile)
    elif prune_method == "channel_level":
        mask = get_threshold_and_mask(torch.norm(prune_layer, p=2, dim=(0, 2, 3), keepdim=True), percentile)


    if mask is not None:
        prune_layer = prune_layer * mask.float()

    if vis:
        visualize_tensor(prune_layer, title=title)  # 实现可视化的函数

    return prune_layer

# 使用PyTorch创建一个张量
tensor = torch.rand((3, 10, 4, 5))

# 调用函数进行剪枝
pruned_tensor = prune_conv_layer(tensor, 'fine_grained', '细粒度剪枝',  vis=True)
pruned_tensor = prune_conv_layer(tensor, 'vector_level', 'Vector级别剪枝', vis=True)
pruned_tensor = prune_conv_layer(tensor, 'kernel_level', 'Kernel级别剪枝', vis=True)
pruned_tensor = prune_conv_layer(tensor, 'filter_level', 'Filter级别剪枝', vis=True)
pruned_tensor = prune_conv_layer(tensor, 'channel_level', 'Channel级别剪枝', vis=True)