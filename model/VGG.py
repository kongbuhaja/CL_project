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
    def __init__(self, channel, n_classes, n_blocks, muls, strides, image_size, in_channel, bn=True, activation='relu'):
        super().__init__()
        flat_channel = np.prod(image_size) 
        
        layers = []
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            layers += [VGGBlock(in_channel, channel * mul, n_block, stride=stride, bn=bn, activation=activation)]
            in_channel = layers[-1].channel
            layers += [nn.MaxPool2d((2,2), 2)]

        layers += [nn.AdaptiveAvgPool2d((1,1))]
        layers += [nn.Flatten()]

        layers += [FC_Block(layers[-4].channel, channel * 64, activation)]
        layers += [nn.Dropout(0.5)]
        layers += [FC_Block(layers[-2].channel, channel * 64, activation)]
        layers += [nn.Dropout(0.5)]
        layers += [FC_Block(layers[-2].channel, n_classes)]

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

def VGG16(channel, n_classes, image_size, in_channel):
    return VGG(channel=channel,
               n_classes=n_classes,
               n_blocks=[2, 2, 3, 3, 3], 
               muls=[1, 2, 4, 8, 8],
               strides=[1, 1, 1, 1, 1], 
               image_size=image_size, 
               in_channel=in_channel,
               activation='relu')


    
