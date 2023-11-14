import torch
import torch.nn as nn
from model.common import *

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, channel, kernel_size=3, stride=1, bn=True, activation='relu'):
        super().__init__()

        layers = [conv(in_channel, channel, kernel_size, stride, bias=not bn)]
        layers += [nn.BatchNorm2d(channel)] if bn else []
        layers += [nn.ReLU()]

        layers += [conv(channel, channel, kernel_size, bias=not bn)]
        layers += [nn.BatchNorm2d(channel)] if bn else []

        self.layers = nn.Sequential(*layers)

        if stride != 1:
            shortcut = [conv(in_channel, channel, 1, stride, bias=not bn)]
            shortcut += [nn.BatchNorm2d(channel)] if bn else []
        else:
            shortcut = [nn.Identity()]

        self.shortcut = nn.Sequential(*shortcut)

        if activation=='relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        out = self.layers(x)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.activation(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, channel, n_classes, n_blocks, muls, strides, in_channel):
        super().__init__()

        layers = [conv(in_channel, channel, 7, 2)]
        in_channel = channel * muls[0]

        layers += [nn.MaxPool2d(3,3)]
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            for s in [stride] + [1] * (n_block-1):
                layers += [block(in_channel, channel * mul, stride=s)]
                in_channel = channel * mul

        layers += [nn.AdaptiveAvgPool2d((1,1))]

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_channel, n_classes)

    def forward(self, x):
        batch_size, height, width, channels = x.size()
        out = x.permute(0,3,1,2).contiguous()
        out = self.layers(out)
        out = self.linear(out.view(batch_size, -1))
        return out

def ResNet18(channel, n_classes, in_channel):
    return ResNet(ResBlock, 
                  channel=channel, 
                  n_classes=n_classes, 
                  n_blocks=[2, 2, 2, 2], 
                  muls=[1, 2, 4, 8], 
                  strides=[1, 2, 2, 2], 
                  in_channel=in_channel)