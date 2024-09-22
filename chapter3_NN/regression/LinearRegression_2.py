import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def make_features(x):
    x = x.unsqueeze(1)  # 在第二维上增加一个维度
    features = [x ** i for i in range(1, 4)]  # 生成x的1次方、2次方、3次方的张量列表 [32*1,32*1,32*1]
    return torch.cat(features, 1)  # 沿着第二维拼接这些张量 [32*3]

# x = torch.tensor([1.0, 2.0, 3.0])  # 假设x是一个一维张量
# features = make_features(x)
# print(features)

w_target = torch.FloatTensor([0.5,3,2.4]).unsqueeze(1).cuda()
b_target = torch.FloatTensor([0.9]).cuda()
x_train = []
y_train = []
def f(x):
    return x.mm(w_target) + b_target

def get_batch(batch_size = 32):
    """buidls a batch i.e. (x,f(x)) pair"""
    random = torch.randn(batch_size).cuda()
    x_train.append(random)
    x = make_features(random)  #shape(32,3)
    y = f(x)                                #shape(32,1)
    y_train.append(y)
    return x,y

#define model
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = poly_model().cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


epochs = 0
while True:
    x, y = get_batch()
    output = model(x)
    loss = criterion(output, y)
    print_loss = loss.item()
    print(print_loss)
    optimizer.zero_grad()
    loss.backward( )
    optimizer.step()
    epochs += 1
    if print_loss <2:
        """control precision"""
        break


weight =[ ]
b = []
# 获取模型的参数
for name, param in model.named_parameters():
    if 'weight' in name:
        # 打印权重，注意weight是一个包含所有权重的张量，对于nn.Linear(3, 1)来说，它是一个形状为(1, 3)的张量
        print(f"Weight (w1, w2, w3): {param.cpu().detach().numpy().flatten()}")
        weight.append(param.cpu().detach().numpy().flatten())
    elif 'bias' in name:
        # 打印偏置项
        print(f"Bias (b): {param.cpu().detach().numpy()[0]}")
        b.append(param.cpu().detach().numpy()[0])

    # 注意：由于模型在CUDA上，我们使用.cpu()来将参数移动到CPU上，以便在CPU上进行打印或进一步处理
# 同时，我们使用.detach()来确保我们不会获取到梯度信息（这对于打印或保存参数通常是不必要的）
# w_target = torch.FloatTensor([0.5,3,2.4]).unsqueeze(1).cuda()
# b_target = torch.FloatTensor([0.9]).cuda()

plt.figure(figsize=(8,6))
plt.scatter(x_train[0].cpu().numpy(),y_train[0].cpu().numpy(),color='blue',label='Orginal Data',marker='o')
x = np.linspace(-3,3,1000)
y = weight[0][0]*x + weight[0][1]*x **2+ weight[0][2]*x**3+ b[0]
plt.plot(x,y,color='red',label='Fitting line')
plt.grid(True)
plt.legend()
plt.show()