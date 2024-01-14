# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

def initializer(layers, fn='he_normal'):
    if fn=='he_normal':
        init_fn = nn.init.kaiming_normal_
    elif fn=='he_uniform':
        init_fn = nn.init.kaiming_uniform_
    elif fn=='xavier_normal':
        init_fn = nn.init.xavier_normal_
    elif fn=='xavier_uniform':
        init_fn = nn.init.xavier_uniform_

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            init_fn(layer.weight.data)
            if not layer.bias == None:
                layer.bias.data.fill_(0)
        elif isinstance(layer, nn.Linear):
            init_fn(layer.weight.data)
            if not layer.bias == None:
                layer.bias.data.fill_(0)

def conv(in_planes, out_planes, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=bias)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
                
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, stride=1, bn=True, activation='relu'):
        super().__init__()
        self.channel = channel

        layers = [conv(in_channel, channel, kernel_size, stride, not bn)]
        layers += [nn.BatchNorm2d(channel)] if bn else []

        if activation=='relu':
            layers += [nn.ReLU()]
        elif activation=='leaky':
            layer += [nn.LeakyReLU(0.01)]

        self.layers = nn.Sequential(*layers)

        initializer(self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out
    
class FC_Block(nn.Module):
    def __init__(self, in_channel, channel, activation=''):
        super().__init__()
        self.channel = channel

        layers = [nn.Linear(in_channel, channel)]

        if activation=='relu':
            layers += [nn.ReLU()]

        self.layers = nn.Sequential(*layers)

        initializer(self.layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


