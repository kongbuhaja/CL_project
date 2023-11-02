import torch
import torch.nn as nn
from model.common import *

class DarknetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, bn=True, activate='relu'):
        super().__init__()

        if kernel_size==3:
            layers = [conv3x3(in_planes, planes, stride)]
        elif kernel_size==1:
            layers = [conv1x1(in_planes, planes, stride)]

        if bn:
            layers = [conv(in_planes,planes, kernel_size, stride, False)]
            layers += [nn.BatchNorm2d(planes)]
        else:
            layers = [conv(in_planes,planes, kernel_size, stride, True)]

        if activate=='relu':
            layers += [nn.ReLU()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out
    
class DarknetTiny(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_channels):
        super().__init__()
        self.num_classes = num_classes
        self.in_planes = nf

        self.conv1 = block(input_channels, nf * 1, 1, 3)

        self.conv2 = self._make_layer(block, nf * 1, num_blocks[0], 1) #2

        self.conv3 = self._make_layer(block, nf * 2, num_blocks[1], 1) #3-5

        self.conv4 = self._make_layer(block, nf * 4, num_blocks[2], 1) #6-8

        self.conv5 = self._make_layer(block, nf * 8, num_blocks[3], 1) #9-13

        self.conv6 = self._make_layer(block, nf * 16, num_blocks[4], 1) #14-18
        # self.conv7 = block(nf * 32 * block.expansion, self.num_classes, 1, 1, bn=False, activate=False) #19
        
        self.gavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(nf*block.expansion*32, self.num_classes)

        self.maxpool = nn.MaxPool2d(2,2)

    def _make_layer(self, block, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        kernel_sizes = [3,1] * (num_block // 2) + [3]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            alpha = 2 if kernel_size==3 else 1
            layers.append(block(self.in_planes, int(planes * alpha * block.expansion), stride, kernel_size))
            self.in_planes = int(planes * alpha * block.expansion)
        return nn.Sequential(*layers)


    def forward(self, x):
        batch_size, height, width, channels = x.size()
        out = self.conv1(x.permute(0,3,1,2).contiguous())
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        out = self.maxpool(out)

        out = self.conv6(out)
        
        # out = self.conv7(out)
        out = self.gavgpool(out)
        out = self.linear(out.view(batch_size, -1))
        return out

def DarkNet19(nclasses, nf=20, input_channels=3):
    return DarknetTiny(DarknetBlock, [1, 3, 3, 5, 5], nclasses, nf, input_channels)


    
