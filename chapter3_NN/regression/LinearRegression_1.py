import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],[9.779], [6.182], [7.59], [2.167], [7.0423],[10.791],[5.313],[7.997],[3.1]], dtype=np.float32)
y_train = np.array( [[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366], [2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[1.3]], dtype=np.float32)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    inputs = torch.from_numpy(x_train).cuda()
    targets = torch.from_numpy(y_train).cuda()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[ {} / {}] ] , loss:{}'.format(epoch + 1, num_epochs, loss.item() ) )

model.eval()
predict = model(torch.from_numpy(x_train).cuda())
predict = predict.cpu().detach().numpy()
plt.figure(figsize=(8, 6))  # 设置图形的大小
plt.scatter(x_train, y_train,color='b',marker='o',label='Original Data')
plt.plot(x_train, predict, color='r', label='Fitting Line')
plt.grid(True)  # 显示网格
plt.legend()
plt.show()  # 显示图形




