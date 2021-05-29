import torch.nn as nn


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
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


network_dict = {"mynet": mynet}
