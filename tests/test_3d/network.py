import torch.nn as nn
import torch
from torch import optim

class Down(nn.Module):
    def __init__(self, in_channels, kernel_size=2, stride=2):
        super(Down, self).__init__()
        out_channels = in_channels * 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Up, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=kernel_size, stride=stride, groups=1)
        self.bn = nn.BatchNorm3d(out_channels // 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input, skip):
        out = self.act(self.bn(self.conv(input)))
        out = torch.cat((out, skip), 1)
        return out

class Input(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Input, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out

class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.act1(self.bn1(self.conv1(input)))
        out = self.conv2(out)
        out = self.softmax(out)
        return out

class SegNet(nn.Module):
    # network1
    # def __init__(self, in_channels, out_channels):
    #     super(SegNet, self).__init__()
    #     self.in_block = Input(in_channels, 8)
    #     self.out_block = Output(8, out_channels)
    #
    # def forward(self, input):
    #     out_in = self.in_block(input)
    #     out = self.out_block(out_in)
    #     return out

    # network2
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()
        self.in_block = Input(in_channels, 8)
        self.conv = Down(8)
        self.up = Up(16, 16)
        self.out_block = Output(16, out_channels)

    def forward(self, input):
        out_in = self.in_block(input)
        out = self.conv(out_in)
        up = self.up(out, out_in)
        out = self.out_block(up)
        return out


if __name__ == '__main__':
    #
    label = torch.randint(0, 3, (10, 1, 32, 32, 32))
    net = SegNet(1, 3)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    for i in range(10):
        data = torch.randn(1, 1, 32, 32, 32)
        optimizer.zero_grad()
        out = net.forward(data)
        l = loss(out, label[i])
        l.backward()
        optimizer.step()
        if i == 9:
            torch.save({'state_dict': net.state_dict()}, 'state_params.pth.tar')