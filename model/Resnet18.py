import torch
import torch.nn as nn
from model.common import *

class Resblock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, bn=True, activate='relu'):
        super().__init__()
        
        layers = [conv(in_planes, planes, kernel_size, stride, bias=False)]
        layers += [nn.BatchNorm2d(planes)]

        layers += [conv(planes, planes, kernel_size, bias=False)]
        layers += [nn.BatchNorm2d(planes)]

        self.layers = nn.Sequential(*layers)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut = nn.Sequential()

        if activate=='relu':
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Identity()
    def forward(self, x):
        out = self.layers(x)
        s = self.shortcut(x)
        out += s
        out = self.activate(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_channels):
        super().__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(input_channels, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.flatten = nn.Flatten()
        self.avg_pool = nn.modules.AvgPool2d(2)
        self.linear = nn.Linear(nf * 8 * 16* block.expansion, num_classes)#nn.Linear(nf * 8 * block.expansion, num_classes)

        self.activate = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes * block.expansion, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # batch_size, height, width, channels = x.size()
        # out = x.view(batch_size, channels, height, width)
        out = self.bn1(self.conv1(x.permute(0,3,1,2).contiguous()))
        out = self.activate(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

def ResNet18(nclasses, nf=20, input_channels=3):
    return ResNet(Resblock, [2, 2, 2, 2], nclasses, nf, input_channels)