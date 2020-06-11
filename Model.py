import tensorflow as tf
import torch
import torch.nn as nn
from torch.autograd import Variable
import Utils
import numpy as np

from NetworkLayers import ResNet, FPN, RPN, FPNClassifier, FPNMask
from FunctionLayers import ProposalLayer, DetectionTargetLayer
import LossFunction
from artifical_data import _GenerateCircle, generate_data
import DataLoader

import sys

print(sys.)

# Mask R-CNN
class MRCNN(nn.Module):
    def __init__(self, in_channels, image_shape, mode, gt_class_ids=None, gt_boxes=None, gt_masks=None):
        super().__init__()
        # TODO: parameter "training" of batchnorm need to be set. (May be some batchnorm hasn't be written.)
        self.resnet = ResNet(in_channels, 2048)
        self.fpn = FPN(2048, 256)
        self.rpn = RPN(256)

        # in channel = fpn out channel, out channel = num_classes
        self.fpn_classifier = FPNClassifier(256, 3, fc_layers_size=1024, pool_size=7, image_shape=image_shape)
        self.fpn_mask = FPNMask(256, 3, mask_pool_size=14, image_shape=image_shape)

        self.mode = mode
        # TODO: image_shape should only be used in training.
        self.image_shape = image_shape
        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks

        # Get when forwarding.
        self.anchors = None  # (batch, n_anchors, 4). Same among batch so don't need batch dim.
        self.rpn_match = None  #
        self.rpn_bbox = None

    def forward(self, x):
        # 1. Backbone
        resnet_feature_maps = self.resnet(x)
        # 2. FPN
        p2, p3, p4, p5, p6 = self.fpn(resnet_feature_maps)
        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        # 3. RPN
        layer_outputs = []
        for p in rpn_feature_maps:
            # layer_outputs [[logits1, probs1, bbox1], [logits2, probs2, bbox2], ...]
            layer_outputs.append(self.rpn(p))

        # outputs: [[logits1, logits2, ...], [probs1, probs2, ...], [bbox1, bbox2, ...]]
        outputs = list(zip(*layer_outputs))
        # Concatenate the logits of different pyramid feature maps on axis 1 to get [logits_all, probs_all, bbox_all].
        # e.g. every logits's shape is (batch, n_anchors, 2).
        rpn_logits, rpn_scores, rpn_bboxes = [torch.cat(x, dim=1) for x in outputs]

        # pyramid shape is [batch, channels, h, w]. So take last 2 as shape.
        pyramid_shapes = [[x.shape[-2], x.shape[-1]] for x in rpn_feature_maps]
        # TODO: It should be confirmed.
        feature_strides = [4, 8, 16, 32, 64]
        # generate anchors
        anchors = self.generate_anchors(pyramid_shapes, feature_strides, p2.shape[0])

        # 4. Proposal Layer
        PL = ProposalLayer(proposal_count=1000)
        rpn_rois = PL.process(anchors, rpn_scores, rpn_bboxes)

        if self.mode == 'train':
            # TODO: Should reduce train_proposals_per_image to around 200. Set to 500 to avoid error when debugging.
            # DetectionTargetLayer
            DTL = DetectionTargetLayer(self.gt_class_ids, self.gt_boxes, self.gt_masks,
                                       proposal_positive_ratio=0.33, train_proposals_per_image=500, mask_shape=[28, 28])
            rois, target_class_ids, target_bbox, target_mask = DTL.process(rpn_rois)

            # Network heads
            # (batch, n_rois, n_classes), (batch, n_rois, n_classes), (batch, n_rois, n_classes, 4)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(rois, mrcnn_feature_maps)
            # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
            mrcnn_mask = self.fpn_mask(rois, mrcnn_feature_maps)
            return rpn_logits, rpn_scores, rpn_bboxes, \
                   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
                   target_class_ids, target_bbox, target_mask

    def generate_anchors(self, pyramid_shapes, feature_strides, batch_size):
        """
        pyramid_shapes: [p2.shape, ...]. Maybe [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feature_strides: Default to be [4, 8, 16, 32, 64].
        batch_size: int.

        return: (batch, n_anchors, 4)
        """
        anchor_genertor = Utils.AnchorGenerator(scales=[32, 64, 96], ratios=[0.5, 1, 1])
        anchors = anchor_genertor.get_anchors(pyramid_shapes, feature_strides)
        anchors = Utils.norm_boxes(anchors, image_shape=self.image_shape)
        # [anchor_counts, 4] to [batch, anchor_counts, 4]
        anchors = np.broadcast_to(anchors, (batch_size,) + anchors.shape)
        anchors = torch.tensor(anchors, dtype=torch.float32)
        self.anchors = anchors
        return anchors

    def get_rpn_targets(self, rpn_train_anchors_per_image=256):
        """
        Get rpn targets and save.
        rpn_train_anchors_per_image: int.

        return:
            rpn_match: (batch, n_anchors). Matches between anchors and GT boxes.
                       1 = positive anchor, -1 = negative anchor, 0 = neutral
            rpn_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)]). Anchor bbox deltas.
        """
        # add batch dim
        rpn_train_anchors_per_image = [rpn_train_anchors_per_image] * self.anchors.shape[0]
        self.rpn_match, self.rpn_bbox = Utils.batch_slice(
            [self.anchors, gt_boxes, rpn_train_anchors_per_image], DataLoader.build_rpn_targets)
        return self.rpn_match, self.rpn_bbox


if __name__ == '__main__':
    # images = torch.rand([10, 1, 256, 256])
    # # 3 instances per image.
    # gt_class_ids = torch.rand([10, 3])
    # gt_boxes = torch.rand([10, 3, 4])
    # gt_masks = torch.rand([10, 3, 256, 256])

    # Get data
    images, gt_class_ids, gt_boxes, gt_masks = generate_data()
    images = torch.tensor(images, dtype=torch.float32)  # (batch, n_classes, 256, 256)
    images = Variable(images)
    gt_class_ids = torch.tensor(gt_class_ids, dtype=torch.float32)  # (batch, n_classes)
    gt_class_ids = Variable(gt_class_ids)
    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)  # (batch, n_classes, 4)
    gt_boxes = Variable(gt_boxes)
    gt_masks = torch.tensor(gt_masks, dtype=torch.float32)  # (batch, n_classes, 256, 256)
    gt_masks = Variable(gt_masks)
    # get active_class_ids
    active_class_ids = torch.where(gt_class_ids == 0,
                                   torch.tensor(0, dtype=torch.int32), torch.tensor(1, dtype=torch.int32))

    mrcnn = MRCNN(1, [256, 256], mode='train', gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks)
    mrcnn.cuda()
    optimizer = torch.optim.Adam(mrcnn.parameters())

    for epoch in range(10):
        # Model
        rpn_logits, rpn_scores, rpn_bboxes, \
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        target_class_ids, target_bbox, target_mask = mrcnn(images)
        # Get tpn targets. (batch, n_anchors). (batch, n_anchors, [dy, dx, log(dh), log(dw)])
        target_rpn_match, target_rpn_bbox = mrcnn.get_rpn_targets()

        # Loss
        loss = LossFunction.compute_loss(rpn_logits, rpn_bboxes,
                                  target_rpn_match, target_rpn_bbox,
                                  mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                                  target_class_ids, target_bbox, target_mask,
                                  active_class_ids)

        # optimize
        loss.backward()
        optimizer.step()
        print(loss)


