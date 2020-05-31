import torch
import torch.nn as nn


# ResNet
class ResBlock(nn.Module):
    """
    Construct Residual block used in the ResNet.
    """
    def __init__(self, in_channels, filters, stride=1, res_conv=False):
        """
        in_channels: the channel number of input tensor
        filters: [n_filter1, n_filter2, n_filter3], the filter number of the three conv blocks
        stride: the stride of the first conv1x1 (including shortcut)
        res_conv: bool, whether conv1x1 is used in the shortcut
        """
        super().__init__()
        self.res_conv = res_conv

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, filters[0], kernel_size=1, stride=stride),
                                   nn.BatchNorm2d(filters[0]),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
                                   nn.BatchNorm2d(filters[1]),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(filters[1], filters[2], kernel_size=1),
                                   nn.BatchNorm2d(filters[2]))
        self.conv1x1 = nn.Conv2d(in_channels, filters[2], kernel_size=1, stride=stride)
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
    Construct ResNet.
    Return: [C1, C2, C3, C4, C5]. feature maps with different depth
    """
    def __init__(self, in_channels, out_channels):
        """
        in_channels: the channel number of the input tensor.
        out_channels: the channel number of the output tensor. Then the filter numbers of the 5 stages will be
                      out_channels/32, out_channels/8, out_channels/4, out_channels/2, out_channels respectively
                      ('//' is used to avoid errors raised because of the float type). It's 2048 in Mask R-CNN.
        """
        super().__init__()
        # stage 1
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels, out_channels//32, kernel_size=7, stride=2, padding=3),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # stage 2
        self.stage2 = self.construct_stage(out_channels//32, out_channels//8, conv_stride=(1, 1), n_blocks=2)
        # stage 3
        self.stage3 = self.construct_stage(out_channels//8, out_channels//4, conv_stride=(2, 2), n_blocks=3)
        # stage 4
        self.stage4 = self.construct_stage(out_channels//4, out_channels//2, conv_stride=(2, 2), n_blocks=5)
        # stage 5
        # block_num should be 5 for resnet50 and 22 for resnet101.
        self.stage5 = self.construct_stage(out_channels//2, out_channels, conv_stride=(2, 2), n_blocks=2)

    def construct_stage(self, in_channels, out_channels, conv_stride, n_blocks):
        """
        n_channels: the channel number of the input tensor
        out_channels: the channel number of the output tensor. Then the filter numbers of the 3 convs will be
                      out_channels/4, out_channels/4, out_channels respectively.
        conv_stride: the stride of the con1x1 in the conv_block
        n_blocks: the number of identity blocks.

        return: a sequence of the blocks
        """
        # add identity block
        layers = [ResBlock(in_channels, [out_channels//4, out_channels//4, out_channels],
                           stride=conv_stride, res_conv=True)]

        # add conv blocks
        for i in range(n_blocks):
            layers.append(ResBlock(out_channels, [out_channels//4, out_channels//4, out_channels]))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c1, c2, c3, c4, c5]


# FPN
class FPN(nn.Module):
    """
    Construct Feature Pyramid Network.
    input: [c1, c2, c3, c4, c5]
    return: [p2, p3, p4, p5, p6]
    """
    def __init__(self, in_channels, out_channels):
        """
        in_channels: the channel number of the input tensor. Default to be 2048 for c5. So it'll be 1024, 512, 156 for
                    c4, c3, c2 respectively.
        out_channels: the channel number of the output tensor. It's 256 in Mask R-CNN.
        """
        super().__init__()
        self.conv2_1 = nn.Conv2d(in_channels//8, out_channels, kernel_size=1)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        self.conv3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels//2, out_channels, kernel_size=1)
        self.conv4_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.conv5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, featuremaps):
        _, c2, c3, c4, c5 = featuremaps
        p5 = self.conv5_1(c5)
        p4 = self.conv4_1(c4)
        p4up = nn.Upsample(scale_factor=2)(p5)
        # p4 = self.conv4_1(c4) + nn.Upsample(scale_factor=2)(p5)
        p3 = self.conv3_1(c3) + nn.Upsample(scale_factor=2)(p4)
        p2 = self.conv2_1(c2) + nn.Upsample(scale_factor=2)(p3)

        p2 = self.conv2_2(p2)
        p3 = self.conv3_2(p3)
        p4 = self.conv4_2(p4)
        p5 = self.conv5_2(p5)
        p6 = self.maxpool5(p5)

        return [p2, p3, p4, p5, p6]


# RPN
class RPN(nn.Module):
    """
    Construct Region Proposal Network.
    input: (batch, channels, height, width), a feature map from P2 to P6 in turn
    return: [rpn_class_logits, rpn_probs, rpn_bbox], the score and delta of every anchors
    """
    def __init__(self, in_channels, anchor_stride=1, anchors_per_location=9):
        """
        in_channels: the channel number of the input tensor
        anchor_stride: the stride of the pixel to get anchors. Default to be 1.
        anchors_per_location: the number of anchors gotten from one pixel. Default to be 9.
        """
        super().__init__()
        self.conv_share = nn.Sequential(nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, stride=anchor_stride),
                                        nn.ReLU())
        self.conv_class = nn.Conv2d(512, 2*anchors_per_location, kernel_size=1)
        self.conv_bbox = nn.Conv2d(512, 4*anchors_per_location, kernel_size=1)

    def forward(self, x):
        """
        Returns:
            rpn_class_logits: (batch, H * W * anchors_per_location, 2) Anchor classifier logits (before softmax)
            rpn_probs: (batch, H * W * anchors_per_location, 2) Anchor classifier probabilities.
            rpn_bbox: (batch, H * W * anchors_per_location, 4) Deltas (dy, dx, log(dh), log(dw))
                     to be applied to anchors.
            H will be H/anchor_stride is anchor_stride is not 1. Same to W.
        """
        shared = self.conv_share(x)

        # Anchor Score. (batch, height, width, anchors per location * 2)
        rpn_class_logits = self.conv_class(shared)
        # Reshape to [batch, anchors, 2]
        rpn_class_logits = rpn_class_logits.reshape([rpn_class_logits.shape[0], -1, 2])
        # Softmax on last dimension of BG/FG.
        rpn_probs = nn.functional.softmax(rpn_class_logits)

        # Bounding box refinement delta. (batch, height, width, anchors per location * 4)
        # the four elements means [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(shared)
        # Reshape to (batch, anchors, 4).
        # The anchors is arranged in order of anchor_n, transverse, longitude. More information in Note.docx
        rpn_bbox = rpn_bbox.reshape([rpn_bbox.shape[0], -1, 4])
        return [rpn_class_logits, rpn_probs, rpn_bbox]