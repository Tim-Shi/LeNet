import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        # 池化层
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 展平操作
        self.flatten = nn.Flatten()
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))





