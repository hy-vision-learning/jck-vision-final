import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from model.shakedrop import ShakeDrop


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, shake=False, p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        
        self.shake = shake
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = self.__make_conv3x3__(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = self.__make_conv3x3__(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shake_drop = ShakeDrop(p_shakedrop)
        self.downsample = downsample
        
    def __make_conv3x3__(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        out = self.relu(self.bn2(self.conv1(self.bn1(x))))
        out = self.bn3(self.conv2(out))
        
        if self.shake:
            out = self.shake_drop(out)
    
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        # padding
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out = out + torch.cat((shortcut, padding), 1)
        else:
            out = out + shortcut 

        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, shake=False, p_shakedrop=1.0):
        super(Bottleneck, self).__init__()
        
        self.shake = shake
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = self.__make_conv1x1__(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = self.__make_conv3x3__(planes, planes, stride=stride)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = self.__make_conv1x1__(planes, planes * Bottleneck.outchannel_ratio)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.shake_drop = ShakeDrop(p_shakedrop)
        self.downsample = downsample

    def __make_conv3x3__(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def __make_conv1x1__(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out = self.bn4(out)
        
        if self.shake:
            out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        # padding
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out = out + torch.cat((shortcut, padding), 1)
        else:
            out = out + shortcut 

        return out


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, num_classes, shake=False, bottleneck=False):
        super(PyramidNet, self).__init__()
        self.inplanes = 16
        self.shake = shake
        
        if bottleneck == True:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock
        
        if self.shake:
            all_depth = n * 3
            self.p_drop = [0.5/all_depth * (i + 1) for i in range(all_depth)]
            self.shake_idx = 0

        self.addrate = alpha / (3 * n * 1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = self.__make_conv3x3__(3, self.input_featuremap_dim)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.__make_layer__(block, n)
        self.layer2 = self.__make_layer__(block, n, stride=2)
        self.layer3 = self.__make_layer__(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __make_layer__(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample,
                            shake=self.shake, p_shakedrop=self.p_drop[self.shake_idx] if self.shake else 0))
        
        if self.shake:
            self.shake_idx += 1
        for _ in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)),
                                1, shake=self.shake, p_shakedrop=self.p_drop[self.shake_idx] if self.shake else 0))
            self.featuremap_dim  = temp_featuremap_dim
            if self.shake:
                self.shake_idx += 1
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def __make_conv3x3__(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
    
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
        
    
def pyramidnet100_84(class_num, shake=False):
    return PyramidNet(100, 84, class_num, shake=shake, bottleneck=True)

def pyramidnet200_240(class_num, shake=False):
    return PyramidNet(200, 240, class_num, shake=shake, bottleneck=True)

def pyramidnet236_220(class_num, shake=False):
    return PyramidNet(236, 220, class_num, shake=shake, bottleneck=True)

def pyramidnet272_200(class_num, shake=False):
    return PyramidNet(272, 200, class_num, shake=shake, bottleneck=True)

def pyramidnet_custom(class_num, depth, alpha, shake=False):
    return PyramidNet(depth, alpha, class_num, shake=shake, bottleneck=True)
    