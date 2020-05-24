import tensorflow as tf
import torch
import torch.nn as nn


# ResNet
class ResBlock(nn.Module):
    """
    residual block used in the ResNet
    """
    def __init__(self, in_channels, filters, stride=(1, 1), res_conv=False):
        """
        in_channels: the channel number of input tensor
        filters: [n_filter1, n_filter2, n_filter3], the filter number of the three conv blocks
        stride: the stride of the first conv1x1 (including shortcut)
        res_conv: bool, whether conv1x1 is used in the shortcut
        """
        super().__init__()
        self.res_conv = res_conv

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, filters[1], kernel_size=1, stride=stride),
                                   nn.BatchNorm2d(filters[1]),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
                                   nn.BatchNorm2d(filters[2]),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(filters[2], filters[3], kernel_size=1),
                                   nn.BatchNorm2d(filters[3]))
        self.conv1x1 = nn.Conv2d(in_channels, filters[3], kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.res_conv:
            x = self.conv1x1(x)
        out += x
        out = self.relu(x)

        return out


class ResNet(nn.Module):
    """
     a ResNet structure.
    Return: [C1, C2, C3, C4, C5]. featuremaps with different depth
    """
    def __init__(self, in_channels, out_channels):
        """
        in_channels: the channel number of the input tensor.
        out_channels: the channel number of the output tensor. Then the filter numbers of the 5 stages will be
                      out_channels/32, out_channels/8, out_channels/4, out_channels/2, out_channels respectively.
                      It's 2048 in Mask R-CNN.
        """
        super().__init__()
        # stage 1
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels, out_channels/32, kernel_size=3, stride=2, padding=3),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # stage 2
        self.stage2 = self.construct_stage(out_channels/32, out_channels/8, conv_stride=(1, 1), n_blocks=2)
        # stage 3
        self.stage2 = self.construct_stage(out_channels/8, out_channels/4, conv_stride=(2, 2), n_blocks=3)
        # stage 4
        self.stage4 = self.construct_stage(out_channels/4, out_channels/2, conv_stride=(2, 2), n_blocks=5)
        # stage 5
        # block_num should be 5 for resnet50 and 22 for resnet101.
        self.stage5 = self.construct_stage(out_channels/2, out_channels, conv_stride=(2, 2), n_blocks=2)

    def construct_stage(self, in_channels, out_channels, conv_stride, n_blocks):
        """

        n_channels: the channel number of the input tensor
        out_channels: the channel number of the output tensor. Then the filter numbers of the 3 convs will be
                      out_channels/4, out_channels/4, out_channels respectively.
        conv_stride: the stride of the con1x1 in the conv_block
        n_blocks: the number of identity blocks.

        return: a sequence of the blocks
        """
        layers = [ResBlock(in_channels, [out_channels/4, out_channels/4, out_channels],
                           stride=conv_stride, res_conv=True)]

        for i in range(n_blocks):
            layers.append(ResBlock(out_channels, [out_channels/4, out_channels/4, out_channels]))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(x)
        c5 = self.stage5(x)
        return [c1, c2, c3,c4, c5]


# FPN
class FPN(nn.Module):
    """
    Construnct Feature Pyramid Network
    input: [c1, c2, c3, c4, c5]
    output: [p2, p3, p4, p5, p6]
    """
    def __init__(self, in_channels, out_channels):
        """
        in_channels: the channel number of the input tensor.
        out_channels: the channel number of the output tensor. It's 256 in Mask R-CNN.
        """
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.conv5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, featuremaps):
        _, c2, c3, c4, c5 = featuremaps
        p5 = self.conv5_1(c5)
        p4 = self.conv4_1(c4) + nn.Upsample(scale_factor=2)(p5)
        p3 = self.conv3_1(c3) + nn.Upsample(scale_factor=2)(p4)
        p2 = self.conv2_1(c2) + nn.Upsample(scale_factor=2)(p3)

        p2 = self.conv2_2(p2)
        p3 = self.conv3_2(p3)
        p4 = self.conv4_2(p4)
        p5 = self.conv5_2(p5)
        p6 = self.maxpool5(p5)

        return [p2, p3, p4, p5, p6]


