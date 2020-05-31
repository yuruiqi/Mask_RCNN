import tensorflow as tf
import torch
import torch.nn as nn
import Utils
import numpy as np

from NetworkLayers import ResNet, FPN, RPN
from FuntionLayers import ProposalLayer


# Mask R-CNN
class MRCNN(nn.Module):
    def __init__(self, in_channels, image_shape):
        super().__init__()
        self.resnet = ResNet(in_channels, 2048)
        self.fpn = FPN(2048, 256)
        self.rpn = RPN(256)
        # TODO: image_shape should only be used in training.
        self.image_shape = image_shape

    def forward(self, x):
        resnet_feature_maps = self.resnet(x)
        p2, p3, p4, p5, p6 = self.fpn(resnet_feature_maps)
        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        layer_outputs = []
        for p in rpn_feature_maps:
            # layer_outputs [[logits1, probs1, bbox1], [logits2, probs2, bbox2], ...]
            layer_outputs.append(self.rpn(p))

        # outputs: [[logits1, logits2, ...], [probs1, probs2, ...], [bbox1, bbox2, ...]]
        outputs = list(zip(*layer_outputs))
        # Concatenate the logits of different pyramid feature maps on axis 1 to get [logits_all, probs_all, bbox_all].
        # e.g. every logits's shape is (batch, n_anchors, 2).
        rpn_logits, rpn_scores, rpn_bboxes = [torch.cat(x, dim=1) for x in outputs]

        # pyramid shape is [batch, channels, h, w]
        pyramid_shapes = [[x.shape[-2], x.shape[-1]] for x in rpn_feature_maps]
        # It should be confirmed.
        feature_strides = [4, 8, 16, 32, 64]
        # generate anchors
        anchor_genertor = Utils.AnchorGenerator(scales=[32, 64, 96], ratios=[0.5, 1, 1])
        anchors = anchor_genertor.get_anchors(pyramid_shapes, feature_strides)
        anchors = Utils.norm_boxes(anchors, image_shape=self.image_shape)
        # [anchor_counts, 4] to [batch, anchor_counts, 4]
        anchors = np.broadcast_to(anchors, (p2.shape[0],)+anchors.shape)
        anchors = torch.from_numpy(anchors)

        # Proposal Layer
        PL = ProposalLayer(proposal_count=1000)
        proposals = PL.process(anchors, rpn_scores, rpn_bboxes)

        return layer_outputs


if __name__ == '__main__':
    x = torch.rand([10, 1, 256, 256])
    mrcnn = MRCNN(1, [256, 256])
    out = mrcnn(x)

