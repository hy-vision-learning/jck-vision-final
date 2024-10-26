import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inner_channel)
        self.conv2 = nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return torch.cat([x, out], 1)
    

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.avg = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.avg(self.conv(self.norm(x)))
    
    
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()
        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]
            
            out_channels = int(reduction * inner_channels)
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        logits = self.linear(output)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def densenet121(class_num):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_class=class_num)

def densenet169(class_num):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_class=class_num)

def densenet201(class_num):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_class=class_num)

def densenet161(class_num):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, num_class=class_num)
