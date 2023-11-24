import torch
import torch.nn as nn
import numpy as np
from model.common import *

class VGGBlock(nn.Module):
    def __init__(self, in_channel, channel, n_layers, kernel_size=3, stride=1, bn=False, activation='relu'):
        super().__init__()
        self.channel = channel        

        layers = []
        for s, c in zip([stride] + [1] * (n_layers-1), [channel] * n_layers):
            layers += [Conv_Block(in_channel, c, kernel_size, s, bn, activation)]
            in_channel = c

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class VGG(nn.Module):
    def __init__(self, channel, n_classes, n_blocks, muls, strides, image_size, in_channel, activation='relu', origin=False):
        super().__init__()
        flat_channel = np.prod(image_size) 
        
        layers = []
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            layers += [VGGBlock(in_channel, channel * mul, n_block, stride=stride, bn=not origin, activation=activation)]
            in_channel = layers[-1].channel
            layers += [nn.MaxPool2d((2,2), 2)]
            flat_channel = flat_channel // 4

        layers += [nn.Flatten()]

        layers += [FC_Block(layers[-3].channel * flat_channel, channel * mul * 4, activation)]
        layers += [FC_Block(layers[-1].channel, channel * mul * 4, activation)]
        layers += [FC_Block(layers[-1].channel, n_classes, activation)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x.permute(0,3,1,2).contiguous()
        out = self.layers(out)
        return out

def VGG19(channel, n_classes, image_size, in_channel):
    return VGG(channel=channel,
               n_classes=n_classes,
               n_blocks=[2, 2, 4, 4, 4], 
               muls=[1, 2, 4, 8, 8],
               strides=[1, 1, 1, 1, 1], 
               image_size=image_size, 
               in_channel=in_channel,
               activation='relu')


    
