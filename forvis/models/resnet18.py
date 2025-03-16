from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (BaseFairseqModel, register_model,
                            register_model_architecture)
from torch import Tensor, nn
from torch.nn import functional as F


# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# 定义 ResNet18 模型
@register_model("resnet18")
class ResNet18(BaseFairseqModel):

    @classmethod
    def build_model(cls, args, task):
        num_classes = len(task.organ_dict)
        return cls(num_classes)

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 初始卷积层：输入 512×512，输出尺寸：(512 + 2*3 - 7)/2 + 1 = 256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化：输出尺寸：(256 + 2*1 - 3)/2 + 1 = 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义四个残差层，层数固定为 [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 自适应平均池化调整特征图尺寸为 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入 x 的尺寸为 [batch_size, 3, 512, 512]
        x = self.conv1(x)  # 输出: [batch_size, 64, 256, 256]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 输出: [batch_size, 64, 128, 128]

        x = self.layer1(x)  # 输出: [batch_size, 64, 128, 128]
        x = self.layer2(x)  # 输出: [batch_size, 128, 64, 64]
        x = self.layer3(x)  # 输出: [batch_size, 256, 32, 32]
        x = self.layer4(x)  # 输出: [batch_size, 512, 16, 16]

        x = self.avgpool(x)  # 输出: [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)  # 展平为 [batch_size, 512]
        x = self.fc(x)  # 全连接层输出
        return x


# 构造一个 ResNet18 网络（共 2+2+2+2 个基本块）
@register_model_architecture("resnet18", "resnet18-512")
def register_model_architecture_resnet18_512(args):
    pass
