import torch
import torch.nn as nn
from model.common import *
from torch.nn.functional import relu

class DarknetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, bn=True):
        super(DarknetBlock, self).__init__()
        if kernel_size==3:
            self.conv = conv3x3(in_planes, planes, stride)
        elif kernel_size==1:
            self.conv = conv1x1(in_planes, planes, stride)
        if bn:
            self.bn = nn.BatchNorm2d(planes)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        out = relu(self.bn(self.conv(x)))
        return out
    
class DarknetTiny(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_channels):
        super(DarknetTiny, self).__init__()
        self.num_classes = num_classes
        self.in_planes = nf

        self.conv1 = block(input_channels, nf * 1, 1, 3)
        
        self.conv2 = self._make_layer(block, nf * 1, num_blocks[0], 1) #2
        
        self.conv3 = self._make_layer(block, nf * 2, num_blocks[0], 1) #3-5

        self.conv4 = self._make_layer(block, nf * 3, num_blocks[1], 1) #6-8

        self.conv5 = self._make_layer(block, nf * 4, num_blocks[2], 1) #9-13

        self.conv6 = self._make_layer(block, nf * 5, num_blocks[3], 1) #14-18
        self.conv7 = block(nf * 5 * block.expansion, self.num_classes, 1, 1) #19

        self.gavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        kernel_sizes = [3,1] * (num_block // 2) + [3]
        
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(block(self.in_planes, planes, stride, kernel_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        # batch_size, height, width, channels = x.size()
        # out = self.conv1(x.view(batch_size, channels, height, width))
        out = self.conv1(x.permute(0,3,1,2).contiguous())
        out = self.maxpool(out)

        out = self.conv2(out)
        # out = self.maxpool(out)

        out = self.conv3(out)
        # out = self.maxpool(out)

        out = self.conv4(out)
        out = self.maxpool(out)

        out = self.conv5(out)
        # out = self.maxpool(out)

        out = self.conv6(out)
        out = self.conv7(out)

        out = self.gavgpool(out)
        return out.view(-1, self.num_classes)

def DarkNet19(nclasses, nf=20, input_channels=3):
    return DarknetTiny(DarknetBlock, [1, 3, 3, 5, 5], nclasses, nf, input_channels)


    
