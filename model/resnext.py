from torch import nn
from torch.nn import functional as F
    
    
class Bottleneck(nn.Module):
    expansion = 4
    Cardinality = 32
    Basewidth = 64
    Depth = 4
    
    def __init__(self, input_channels, output_channels, cardinality, stride=1):
        C = Bottleneck.Cardinality
        D = int(Bottleneck.Depth * output_channels / Bottleneck.Basewidth)
        
        super(Bottleneck, self).__init__()
        self.conv1 = self.__make_conv1x1__(input_channels, C * D)
        self.bn1 = nn.BatchNorm2d(C * D)
        self.conv2 = self.__make_conv3x3__(C * D, C * D, stride=stride, groups=Bottleneck.Cardinality)
        self.bn2 = nn.BatchNorm2d(C * D)
        self.conv3 = self.__make_conv1x1__(C * D, output_channels * self.expansion)
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
    
    def __make_conv3x3__(self, input_channels, output_channels, stride=1, padding=1, groups=1):
        return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False)
    
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = x + self.shortcut(identity)
        x = self.relu(x)
        
        return x
    
    
class ResNext(nn.Module):
    def __init__(self, num_classes, cardinality=32):
        super(ResNext, self).__init__()
        self.input_channels = 64
        self.cardinality = cardinality
        
        conv1 = nn.Conv2d(3, self.input_channels, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            conv1,
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU()
        )

        self.conv2 = self._make_layer(64, 3, downsize=False)
        self.conv3 = self._make_layer(128, 4)
        self.conv4 = self._make_layer(256, 6)
        self.conv5 = self._make_layer(512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channel, num_blocks, downsize=True):
        strides = [2 if downsize else 1] + [1] * (num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(Bottleneck(self.input_channels, out_channel, self.cardinality, strides[i]))
            self.input_channels = out_channel * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    
    
def resnext50(class_num):
    return ResNext(class_num, 32)
