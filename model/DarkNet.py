import torch
import torch.nn as nn
from model.common import *
    
class DarkNetBlock(nn.Module):
    def __init__(self, in_channel, channel, n_layers, kernel_size=3, stride=1, bn=True, activation='relu'):
        super().__init__()

        layers = []
        for s, k in zip([stride] + [1] * (n_layers-1), [kernel_size, 1] * (n_layers//2) + [kernel_size]):
            c = channel // 1 if k==3 else channel // 2
            layers += [Conv_Block(in_channel, c, k, s, bn, activation)]
            in_channel = c

        self.layers = nn.Sequential(*layers)
        self.channel = in_channel
        
    def forward(self, x):
        out = self.layers(x)
        return out

class DarkNetTiny(nn.Module):
    def __init__(self, channel, n_classes, n_blocks, muls, strides, in_channel, activation):
        super().__init__()
        
        layers = []
        for n_block, mul, stride in zip(n_blocks[:-1], muls[:-1], strides[:-1]):
            layers += [DarkNetBlock(in_channel, channel * mul, n_block, stride=stride, activation=activation)]
            in_channel = layers[-1].channel
            layers += [nn.MaxPool2d((2,2), 2)]
        layers += [DarkNetBlock(in_channel, channel * muls[-1], n_blocks[-1], stride=strides[-1], activation=activation)]
        in_channel = layers[-1].channel

        layers += [Conv_Block(in_channel, n_classes, 1, 1, False, activation)]
        layers += [nn.AdaptiveAvgPool2d((1,1))]
        layers += [nn.Flatten()]

        # layers += [nn.AdaptiveAvgPool2d((1,1))]
        # layers += [nn.Flatten()]
        # layers += [FC_Block(in_channel, n_classes)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x.permute(0,3,1,2).contiguous()
        out = self.layers(out)
        return out

def DarkNet19(channel, n_classes, in_channel):
    return DarkNetTiny(channel=channel//2, 
                       n_classes=n_classes, 
                       n_blocks=[1, 1, 3, 3, 5, 5], 
                       muls=[1, 2, 4, 8, 16, 32], 
                       strides=[1, 1, 1, 1, 1, 1], 
                       in_channel=in_channel, 
                       activation='leaky')


    
