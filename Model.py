import tensorflow as tf
import torch
import torch.nn as nn
import Utils
import numpy as np

from NetworkLayers import ResNet, FPN, RPN, FPNClassifier, FPNMask
from FunctionLayers import ProposalLayer, DetectionTargetLayer
from LossFunction import compute_class_loss
from artifical_data import _GenerateCircle


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
        self.fpn_mask = FPNMask(256, 3, pool_size=7, image_shape=image_shape)

        self.mode = mode
        # TODO: image_shape should only be used in training.
        self.image_shape = image_shape
        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks

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
            return mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask


if __name__ == '__main__':
    # images = torch.rand([10, 1, 256, 256])
    # # 3 instances per image.
    # gt_class_ids = torch.rand([10, 3])
    # gt_boxes = torch.rand([10, 3, 4])
    # gt_masks = torch.rand([10, 3, 256, 256])

    images, gt_class_ids, gt_boxes, gt_masks = _GenerateCircle()
    images = torch.tensor(images, dtype=torch.float32)
    gt_class_ids = torch.tensor(gt_class_ids, dtype=torch.float32)
    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
    gt_masks = torch.tensor(gt_masks, dtype=torch.float32)

    mrcnn = MRCNN(1, [256, 256], mode='train', gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks)
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask = mrcnn(images)
    print(mrcnn_class_logits)
    # Loss
    # rpn_class_loss = compute_class_loss()

