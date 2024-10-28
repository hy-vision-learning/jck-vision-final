from torch import nn
from torch.nn import functional as F


class WideBasic(nn.Module):
    def __init__(self, input_channels, output_channels, drop_p, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv1 = self.__make_conv3x3__(input_channels, output_channels, stride=1)
        self.dropout = nn.Dropout(drop_p)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv2 = self.__make_conv3x3__(output_channels, output_channels, stride=stride)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                self.__make_conv1x1__(input_channels, output_channels, stride=stride),
                nn.BatchNorm2d(output_channels)
            )
    
    def __make_conv1x1__(self, input_channels, output_channels, stride=1):
        return nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)
    
    def __make_conv3x3__(self, input_channels, output_channels, stride=1, padding=1, groups=1):
        return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False)
    
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(x))
        x = self.dropout(self.conv1(x))
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        
        return x + self.shortcut(identity)
    

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth, k, drop_p):
        super(WideResNet, self).__init__()
        self.input_channels = 16
        
        assert (depth - 4) % 6 == 0, "depth error"
        n = (depth - 4) // 6

        self.conv1 = self.__make_conv3x3__(3, 16)
        self.conv2 = self._make_layer(16 * k, n, drop_p, downsize=False)
        self.conv3 = self._make_layer(32 * k, n, drop_p)
        self.conv4 = self._make_layer(64 * k, n, drop_p)
        self.bn1 = nn.BatchNorm2d(64 * k, momentum=0.9)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * k, num_classes)
        
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, out_channel, num_blocks, drop_p, downsize=True):
        strides = [2 if downsize else 1] + [1] * (num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(WideBasic(self.input_channels, out_channel, drop_p, strides[i]))
            self.input_channels = out_channel
        return nn.Sequential(*layers)
    
    def __make_conv3x3__(self, input_channels, output_channels, stride=1, padding=1, groups=1):
        return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    

def wide_resnet_16_4(class_num):
    return WideResNet(class_num, 16, 4, 0)
    

def wide_resnet_16_10(class_num):
    return WideResNet(class_num, 16, 10, 0)


def wide_resnet_28_10_03(class_num):
    return WideResNet(class_num, 28, 10, 0.3)