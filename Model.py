import tensorflow as tf
import torch
import torch.nn as nn
import Utils
import numpy as np

from NetworkLayers import ResNet, FPN, RPN
from FuntionLayers import ProposalLayer, DetectionTargetLayer


# Mask R-CNN
class MRCNN(nn.Module):
    def __init__(self, in_channels, image_shape, mode, gt_class_ids=None, gt_boxes=None, gt_masks=None):
        super().__init__()
        self.resnet = ResNet(in_channels, 2048)
        self.fpn = FPN(2048, 256)
        self.rpn = RPN(256)
        self.mode = mode
        # TODO: image_shape should only be used in training.
        self.image_shape = image_shape
        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks

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
        anchors = torch.tensor(anchors, dtype=torch.float32)

        # Proposal Layer
        PL = ProposalLayer(proposal_count=1000)
        rpn_rois = PL.process(anchors, rpn_scores, rpn_bboxes)

        if self.mode == 'train':
            self.train_branch(rpn_rois)

        return layer_outputs

    def train_branch(self, rpn_rois):
        DTL = DetectionTargetLayer(self.gt_class_ids, self.gt_boxes, self.gt_masks,
                                   proposal_positive_ratio=0.33, train_proposals_per_image=200)
        rois, target_class_ids, target_bbox, target_mask = DTL.process(rpn_rois)


if __name__ == '__main__':
    images = torch.rand([10, 1, 256, 256])
    gt_class_ids = torch.rand([10, 3])
    gt_boxes = torch.rand([10, 3, 4])
    gt_masks = torch.rand([10, 3, 256, 256])

    mrcnn = MRCNN(1, [256, 256], mode='train', gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks)
    out = mrcnn(images)

