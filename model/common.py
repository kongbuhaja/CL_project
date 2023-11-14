# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

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
    def __init__(self, in_channel, channel, kernel_size=3, stride=1, bn=False, activation='relu'):
        super().__init__()

        layers = [conv(in_channel, channel, kernel_size, stride, not bn)]
        layers += [nn.BatchNorm2d(channel)] if bn else []

        if activation=='relu':
            layers += [nn.ReLU()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out
    
class FC_Block(nn.Module):
    def __init__(self, in_channel, channel, activation):
        super().__init__()

        layers = [nn.Linear(in_channel, channel)]

        if activation=='relu':
            layers += [nn.ReLU()]

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out




