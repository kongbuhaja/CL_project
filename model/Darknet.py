import torch
import torch.nn as nn
from model.common import *
    
class DarknetBlock(nn.Module):
    def __init__(self, in_channel, channel, n_layers, kernel_size=3, stride=1, bn=True, activation='relu'):
        super().__init__()
        self.channel = channel

        layers = []
        for s, k in zip([stride] + [1] * (n_layers-1), [kernel_size, 1] * (n_layers//2) + [kernel_size]):
            c = channel // 1 if k==3 else 2
            layers += [Conv_Block(in_channel, c, k, s, bn, activation)]
            in_channel = c

        self.layers = nn.Sequential(*layers)
        self.out_channel = channel

    def forward(self, x):
        out = self.layers(x)
        return out

class DarknetTiny(nn.Module):
    def __init__(self, block, channel, n_classes, n_blocks, muls, strides, in_channel, activation):
        super().__init__()
        
        layers = []
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            layers += [block(in_channel, channel * mul, n_block, stride=stride, activation=activation)]
            in_channel = layers[-1].channel
            layers += [nn.MaxPool2d(2,2)]
        
        layers += [nn.AdaptiveAvgPool2d((1,1))]

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_channel, n_classes)

    def forward(self, x):
        batch_size, height, width, channels = x.size()
        out = x.permute(0,3,1,2).contiguous()
        out = self.layers(out)
        out = self.linear(out.view(batch_size, -1))
        return out

def DarkNet19(channel, n_classes, in_channel):
    return DarknetTiny(DarknetBlock, 
                       channel=channel, 
                       n_classes=n_classes, 
                       n_blocks=[1, 1, 3, 3, 5, 5], 
                       muls=[1, 2, 4, 8, 16, 32], 
                       strides=[1, 1, 1, 1, 1, 1], 
                       in_channel=in_channel, 
                       activation='relu')


    