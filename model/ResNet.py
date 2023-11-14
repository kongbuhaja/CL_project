import torch
import torch.nn as nn
from model.common import *

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, channel, kernel_size=3, stride=1, bn=True, activation='relu'):
        super().__init__()
        self.channel = channel

        layers = [Conv_Block(in_channel, channel, kernel_size, stride, bn, activation)]
        # layers = [conv(in_channel, channel, kernel_size, stride, bias=not bn)]
        # layers += [nn.BatchNorm2d(channel)] if bn else []
        # layers += [nn.ReLU()]

        layers += [Conv_Block(channel, channel, kernel_size, stride, bn, None)]
        # layers += [conv(channel, channel, kernel_size, bias=not bn)]
        # layers += [nn.BatchNorm2d(channel)] if bn else []

        self.layers = nn.Sequential(*layers)

        if stride != 1:
            self.shortcut = Conv_Block(in_channel, channel, 1, stride, bn, None)
            # shortcut = [conv(in_channel, channel, 1, stride, bias=not bn)]
            # shortcut += [nn.BatchNorm2d(channel)] if bn else []
        else:
            self.shortcut = nn.Identity()

        # self.shortcut = nn.Sequential(*shortcut)

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
    def __init__(self, channel, n_classes, n_blocks, muls, strides, in_channel):
        super().__init__()

        layers = [Conv_Block(in_channel, channel, kernel_size=7, stride=2)]
        in_channel = channel * muls[0]

        layers += [nn.MaxPool2d((3,3), 2)]
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            for s in [stride] + [1] * (n_block-1):
                layers += [ResBlock(in_channel, channel * mul, stride=s)]
                in_channel = layers[-1].channel

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
    return ResNet(channel=channel, 
                  n_classes=n_classes, 
                  n_blocks=[2, 2, 2, 2], 
                  muls=[1, 2, 4, 8], 
                  strides=[1, 2, 2, 2], 
                  in_channel=in_channel)