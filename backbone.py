import torch.nn as nn


class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2,stride=2) 
        self.conv2 = nn.Conv2d(64, 48, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.mp2= nn.MaxPool2d(2,stride=2)
        self.__in_features = 48*4*4

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class dbbhm_net(nn.Module):
    def __init__(self):
        super(dbbhm_net, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.lrelu1 = nn.LeakyReLU()
        self.mp1 = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(64, 50, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(50)
        self.lrelu2 = nn.LeakyReLU()
        self.mp2= nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(50, 50, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(50)
        self.lrelu3 = nn.LeakyReLU()
        self.mp3= nn.MaxPool2d(2)
        self.__in_features = 50*5*5

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.mp3(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


network_dict = {"mnist_net": mnist_net,
                "dbbhm_net": dbbhm_net}
