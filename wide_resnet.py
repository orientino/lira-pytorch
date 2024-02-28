import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, n_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-ResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, stages[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._wide_layer(wide_basic, stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(stages[3], momentum=0.9)
        self.linear = nn.Linear(stages[3], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, n_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(n_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
