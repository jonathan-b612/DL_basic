from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def get_acc(output, label):
    return torch.sum(torch.argmax(output, dim=1) == label).item() /output.shape[0]


def train(net, train_dataloader, test_dataloader, criterion, optimizer, epoch):
    prev_time = datetime.now()
    for epoch in range(epoch):
        train_loss = 0
        train_acc = 0

        for im, label in train_dataloader:
            im = im.cuda()
            label = label.cuda()
            output = net(im)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc +=get_acc(output, label)

        cur_time = datetime.now()
        h,reminder = divmod((cur_time - prev_time).seconds, 3600)
        m,s = divmod(reminder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if test_dataloader is not None:
            test_loss = 0
            test_acc = 0
            net = net.eval()
            for im, label in test_dataloader:
                im = im.cuda()
                label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                test_loss += loss.item()
                test_acc += ( torch.sum(torch.argmax(output, dim=1) == label).item() /output.shape[0] )
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch+1, train_loss / len(train_dataloader),
                       train_acc / len(train_dataloader), test_loss / len(test_dataloader),
                       test_acc / len(test_dataloader)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch+1, train_loss / len(train_dataloader),
                          train_acc / len(train_dataloader)))
        prev_time = cur_time
        print(epoch_str + time_str)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)


class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2), residual_block(64, 64), residual_block(64, 64))

        self.block3 = nn.Sequential(
            residual_block(64, 128, False), residual_block(128, 128))

        self.block4 = nn.Sequential(
            residual_block(128, 256, False), residual_block(256, 256))

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512), nn.AvgPool2d(3))

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x