import torch
import torch.nn as nn
import numpy as np
from model.common import *
from torchvision.models import vgg16_bn, vgg11_bn, vgg19_bn

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
    def __init__(self, channel, n_classes, n_blocks, muls, strides, in_channel, image_size, bn=True, activation='relu'):
        super().__init__()
        self.image_size = np.array(image_size)
        
        layers = []
        for n_block, mul, stride in zip(n_blocks, muls, strides):
            layers += [VGGBlock(in_channel, channel * mul, n_block, stride=stride, bn=bn, activation=activation)]
            in_channel = layers[-1].channel
            layers += [nn.MaxPool2d((2,2), 2)]
            self.image_size //= 2
        
        # layers += [nn.AdaptiveAvgPool2d((7,7))]
        layers += [nn.Flatten()]

        layers += [FC_Block(layers[-3].channel*np.prod(self.image_size), channel * 64, activation)]
        layers += [nn.Dropout(0.5)]
        layers += [FC_Block(layers[-2].channel, channel * 64, activation)]
        layers += [nn.Dropout(0.5)]
        layers += [FC_Block(layers[-2].channel, n_classes)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

def VGG19(channel, n_classes, in_channel, image_size, official):
    if not official:
        model = VGG(channel=channel,
                    n_classes=n_classes,
                    n_blocks=[2, 2, 4, 4, 4], 
                    muls=[1, 2, 4, 8, 8],
                    strides=[1, 1, 1, 1, 1],  
                    in_channel=in_channel,
                    image_size=image_size,
                    activation='relu')
    else:
        model = vgg19_bn()
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)
    return model

def VGG16(channel, n_classes, in_channel, image_size, official):
    if not official:
        model = VGG(channel=channel,
                    n_classes=n_classes,
                    n_blocks=[2, 2, 3, 3, 3], 
                    muls=[1, 2, 4, 8, 8],
                    strides=[1, 1, 1, 1, 1], 
                    in_channel=in_channel,
                    image_size=image_size,
                    activation='relu')
    else:
        model = vgg16_bn()
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)
    return model

def VGG11(channel, n_classes, in_channel, image_size, official):
    if not official:
        model = VGG(channel=channel,
                    n_classes=n_classes,
                    n_blocks=[1, 1, 2, 2, 2], 
                    muls=[1, 2, 4, 8, 8],
                    strides=[1, 1, 1, 1, 1], 
                    in_channel=in_channel,
                    image_size=image_size,
                    activation='relu')
    else:
        print("off")
        model = vgg11_bn()
        model.classifier[-1] = torch.nn.Linear(4096, n_classes)
    return model

    
