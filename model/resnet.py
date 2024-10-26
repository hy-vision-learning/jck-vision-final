from torch import nn
from torch.nn import functional as F


class ResidualCNN(nn.Module):
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualCNN, self).__init__()
        
        conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)

        self.conv = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            conv2,
            nn.BatchNorm2d(out_channel),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Dropout(p=0.2),
            )

    def forward(self, x):
        out = F.relu(self.conv(x) + self.shortcut(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, input_channels, output_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = self.__make_conv1x1__(input_channels, output_channels)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = self.__make_conv3x3__(output_channels, output_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = self.__make_conv1x1__(output_channels, output_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(output_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels * self.expansion:
            self.shortcut = nn.Sequential(
                self.__make_conv1x1__(input_channels, output_channels * self.expansion, stride=stride),
                nn.BatchNorm2d(output_channels * self.expansion)
            )
    
    def __make_conv1x1__(self, input_channels, output_channels, stride=1):
        return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)
    
    def __make_conv3x3__(self, input_channels, output_channels, stride=1, padding=1):
        return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
    
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = x + self.shortcut(identity)
        x = self.relu(x)
        
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.input_channels = 64
        
        conv1 = nn.Conv2d(3, self.input_channels, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU()
        )

        self.conv2 = self._make_layer(block, 64, layers[0], downsize=False)
        self.conv3 = self._make_layer(block, 128, layers[1])
        self.conv4 = self._make_layer(block, 256, layers[2])
        self.conv5 = self._make_layer(block, 512, layers[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, downsize=True):
        strides = [2 if downsize else 1] + [1] * (num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.input_channels, out_channel, strides[i]))
            self.input_channels = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv2(x)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    

def resnet18(class_num):
    return ResNet(class_num, ResidualCNN, [2,2,2,2])
    

def resnet34(class_num):
    return ResNet(class_num, ResidualCNN, [3,4,6,3])
    

def resnet50(class_num):
    return ResNet(class_num, Bottleneck, [3,4,6,3])
    

def resnet101(class_num):
    return ResNet(class_num, Bottleneck, [3,4,23,3])
    

def resnet152(class_num):
    return ResNet(class_num, Bottleneck, [3,8,36,3])
