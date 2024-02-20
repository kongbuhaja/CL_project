import torch
import torch.nn as nn
import numpy as np
from model.common import *
from torchvision.models import googlenet

class Inception(nn.Module):
    def __init__(self, in_channel, channels, bn=True, activation='relu'):
        super().__init__()
        self.channel = np.sum([np.array(channel).reshape((-1,1))[-1] for channel in channels])
        
        self.conv1x1 = Conv_Block(in_channel, channels[0], kernel_size=1, bn=bn, activation=activation)
        
        conv3x3 = [Conv_Block(in_channel, channels[1][0], kernel_size=1, bn=bn, activation=activation)]
        conv3x3 += [Conv_Block(channels[1][0], channels[1][1], kernel_size=3, bn=bn, activation=activation)]
        self.conv3x3 = nn.Sequential(*conv3x3)

        conv5x5 = [Conv_Block(in_channel, channels[2][0], kernel_size=1, bn=bn, activation=activation)]
        conv5x5 += [Conv_Block(channels[2][0], channels[2][1], kernel_size=5, bn=bn, activation=activation)]
        self.conv5x5 = nn.Sequential(*conv5x5)

        max_pool = [nn.MaxPool2d((3,3), 1, padding=1)]
        max_pool += [Conv_Block(in_channel, channels[3], kernel_size=1, bn=bn, activation=activation)]
        self.max_pool = nn.Sequential(*max_pool)

    def forward(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        outmax = self.max_pool(x)
        return torch.concat([out1x1, out3x3, out5x5, outmax], dim=1)
    
class GoogleNet(nn.Module):
    def __init__(self, channel, n_classes, in_channel):
        super().__init__()

        layers = [Conv_Block(in_channel, channel, kernel_size=7, stride=2)]
        layers += [nn.MaxPool2d((3,3), 2, 1)]

        layers += [Conv_Block(layers[-2].channel, channel, kernel_size=1)]
        layers += [Conv_Block(layers[-1].channel, channel*3)]
        layers += [nn.MaxPool2d((3,3), 2, 1)]

        layers += [Inception(layers[-2].channel, [channel, [int(channel*1.5), channel*2], [channel//4, channel//2], channel//2])]
        layers += [Inception(layers[-1].channel, [channel*2, [channel*2, channel*3], [channel//2, int(channel*1.5)], channel])]
        layers += [nn.MaxPool2d((3,3), 2, 1)]

        layers += [Inception(layers[-2].channel, [channel*3, [int(channel*1.5), int(channel*3.25)], [channel//4, int(channel*0.75)], channel])]
        layers += [Inception(layers[-1].channel, [int(channel*2.5), [int(channel*1.75), int(channel*3.5)], [int(channel*0.375), channel], channel])]
        layers += [Inception(layers[-1].channel, [channel*2, [channel*2, channel*4], [int(channel*0.375), channel], channel])]
        layers += [Inception(layers[-1].channel, [int(channel*1.75), [int(channel*2.25), int(channel*4.5)], [channel//2, channel], channel])]
        layers += [Inception(layers[-1].channel, [channel*4, [int(channel*2.5), channel*5], [channel//2, channel*2], channel*2])]
        layers += [nn.MaxPool2d((3,3), 2, 1)]

        layers += [Inception(layers[-2].channel, [channel*4, [int(channel*2.5), channel*5], [channel//2, channel*2], channel*2])]
        layers += [Inception(layers[-1].channel, [channel*6, [channel*3, channel*6], [int(channel*0.75), channel*2], channel*2])]
    
        layers += [nn.AdaptiveAvgPool2d((1,1))]

        layers += [nn.Flatten()]
        layers += [nn.Dropout(0.4)]
        layers += [FC_Block(layers[-4].channel, n_classes)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

def GoogleNet22(channel, n_classes, in_channel, official):
    if not official:
        model = GoogleNet(channel, n_classes, in_channel)
    else:
        model = googlenet(aux_logits=False)
        model.fc = nn.Linear(1024, n_classes)
        
    return model

        
