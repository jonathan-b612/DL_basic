import torch
from torch import nn

# 定义一个单步的 rnn
rnn_single = nn.RNNCell(input_size=100, hidden_size=200)

# # 访问其中的参数
# print(rnn_single.weight_ih.shape)
# print(rnn_single.weight_hh.shape)

# 构造一个序列，长为 6，batch 是 5， 特征是 100
x = torch.randn(6, 5, 100) # 这是 rnn 的输入格式

# 定义初始的记忆状态
h_t = torch.zeros(5, 200)

# 传入 rnn
out = []
''' nn.rnn_cell '''
# for i in range(6): # 通过循环 6 次作用在整个序列上
#     h_t = rnn_single(x[i], h_t)
#     out.append(h_t)

""" nn.rnn"""
rnn_seq = nn.RNN(100, 200)
# 访问其中的参数 x.shape(6,5,100)
out_1,h_t_1 = rnn_seq(x)  # 使用默认的全 0 隐藏状态
# out_1[-1] equal h_t_1
# print(out_1.shape)  # 网络的最后状态(times,batch,features)
# print(h_t_1.shape)  # 隐藏层最终状态(times(last),batch,features)

lstm_seq = nn.LSTM(50, 100, num_layers=2)
lstm_input = torch.randn(10, 3, 50)  # 序列 10，batch 是 3，输入维度 50
out_2,h_t_2 = lstm_seq(lstm_input)  # h_t_2（h,c）,h是未和输出门相乘的输出(不算最终输出) ， c是记忆单元
print(out_2.shape)
print(h_t_2[0].shape,h_t_2[1].shape)
