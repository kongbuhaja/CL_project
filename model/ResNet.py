import torch
import torch.nn as nn
from model.common import *
from torchvision.models import resnet18

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, stride=1, bn=True, activation='relu'):
        super().__init__()
        self.channel = channel
        self.stride = stride

        layers = [Conv_Block(in_channel, channel, kernel_size, stride, bn, activation)]
        layers += [Conv_Block(channel, channel, kernel_size, 1, bn, None)]

        self.layers = nn.Sequential(*layers)

        if stride != 1:
            self.shortcut = Conv_Block(in_channel, channel, 1, stride, bn, None)
        else:
            self.shortcut = nn.Identity()

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
    def __init__(self, channel, n_classes, n_blocks, muls, strides, in_channel, activation):
        super().__init__()

        layers = [Conv_Block(in_channel, channel * muls[0], kernel_size=7, stride=2, activation=activation)]
        in_channel = channel * muls[0]

        layers += [nn.MaxPool2d((3,3), 2, 1)]
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            for s in [stride] + [1] * (n_block-1):
                layers += [ResBlock(in_channel, channel * mul, stride=s, activation=activation)]
                in_channel = layers[-1].channel

        layers += [nn.AdaptiveAvgPool2d((1,1))]
        layers += [nn.Flatten()]
        layers += [FC_Block(in_channel, n_classes)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

def ResNet18(channel, n_classes, in_channel, official=False):
    if not official:
        model = ResNet(channel=channel, 
                       n_classes=n_classes, 
                       n_blocks=[2, 2, 2, 2], 
                       muls=[1, 2, 4, 8], 
                       strides=[1, 2, 2, 2], 
                       in_channel=in_channel,
                       activation='relu')
    else:
        model = resnet18()
        model.fc = nn.Linear(512, n_classes)

    return model