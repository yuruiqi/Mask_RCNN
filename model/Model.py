import torch
import torch.nn as nn

from model import Utils
from model.NetworkLayers import ResNet50, ResNetPyTorch, FPN, RPN, FPNClassifier, FPNMask
from model.FunctionLayers import ProposalLayer, DetectionTargetLayer, DetectionLayer

import numpy as np
import os

grads = Utils.GradSaver()


# Mask R-CNN
class MRCNN(nn.Module):
    def __init__(self, image_shape, n_classes, scales=(32, 64, 128, 256, 512), ratios=(0.5, 1, 2),
                 p4_box_size=224.0, mode='train', pretrain=True):
        # TODO: in_channels
        super().__init__()
        self.mode = mode
        self.image_shape = image_shape
        self.n_classes = n_classes

        # to 256*256 img, box is mostly from 20 to 80.
        # If stride is [4, 8, 16, 32, 64], then hope box 24 apply to P4(stride 16) with feature map size about 3.
        self.anchor_per_location = len(ratios)

        # default for resnet50
        feature_strides = [4, 8, 16, 32, 64]
        # if image size is 256, pyramid shapes is [[64,64], [32,32], [16,16], [8,8], [4,4]]
        pyramid_shapes = [np.array(image_shape)/x for x in feature_strides]
        # anchors = self.anchor_genertor.get_anchors(pyramid_shapes, feature_strides)
        anchors = Utils.generate_pyramid_anchors(scales, ratios, pyramid_shapes, feature_strides, 1)
        self.anchors = Utils.norm_boxes(anchors, image_shape=image_shape)  # (n_anchors, 4)

        self.gt_class_ids = None
        self.gt_boxes = None
        self.gt_masks = None
        self.active_class_ids = None

        # visualization feature map
        self.vfm = {}

        # self.resnet = ResNet50(in_channels, 2048)
        # TODO: channel
        self.resnet = ResNetPyTorch(pretrained=pretrain)
        self.fpn = FPN(2048, 256)
        self.rpn = RPN(256, anchors_per_location=self.anchor_per_location)

        self.proposal_layer = ProposalLayer(post_nms_rois=50, nms_threshold=0.7, pre_nms_limit=100)
        self.detection_target_layer = DetectionTargetLayer(proposal_positive_ratio=0.33, train_proposals_per_image=30,
                                                           mask_shape=[28, 28], positive_iou_threshold=0.5)
        self.detection_layer = DetectionLayer(detection_max_instances=3, detection_nms_threshold=0.3)
        # in channel = fpn out channel, out channel = num_classes
        self.fpn_classifier = FPNClassifier(256, n_classes, fc_layers_size=1024, pool_size=7, image_shape=image_shape,
                                            p4_box_size=p4_box_size)
        self.fpn_mask = FPNMask(256, n_classes, mask_pool_size=14, image_shape=image_shape, p4_box_size=p4_box_size)

    def forward(self, x):
        """
        x: (batch, channels, h, w)

        return:
            TODO: To modify.
            train:
                rpn_logits: (batch, n_anchors, 2)
                rpn_scores: (batch, n_anchors, 2)
                rpn_bbox: (batch, n_anchors, 4)

                mrcnn_class_logits: (batch, n_rois, n_classes)
                mrcnn_class: (batch, n_rois, n_classes)
                mrcnn_bbox: (batch, n_rois, n_classes, [dy, dx, log(dh), log(dw)])

                mrcnn_mask: (batch, num_rois, n_classes, mask_h, mask_w)

                target_class_ids: (batch, n)
                target_bbox: (batch, n, 4)
                target_mask: (batch, n, mask_h, mask_w)

            Inference:
                detection_boxes: (batch, detection_max_instance, [y1, x1, y2, x2])
                detection_classes: (batch, detection_max_instance, [class_id])
                detection_scores: (batch, detection_max_instance, [score])
                mrcnn_masks: (batch, num_rois, n_classes, mask_h, mask_w)
        """
        # 1. Backbone
        resnet_feature_maps = self.resnet(x)

        # 2. FPN
        p2, p3, p4, p5, p6 = self.fpn(resnet_feature_maps)
        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        # self.vfm["rpn_feature_maps"] = rpn_feature_maps
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

        # [anchor_counts, 4] to [batch, anchor_counts, 4]
        anchors = np.broadcast_to(self.anchors, (x.shape[0],) + self.anchors.shape)
        anchors = torch.tensor(anchors, dtype=torch.float32, device=x.device)
        # TODO: should self.anchors have batch dim
        # self.anchors = anchors

        # 4. Proposal Layer
        # print(rpn_scores[...,1].gt(0.5).nonzero().shape[0])
        rpn_rois = self.proposal_layer.process(anchors, rpn_scores, rpn_bboxes)
        self.vfm.update(self.proposal_layer.vfm)
        self.vfm['rpn_rois'] = rpn_rois

        # observer = Observer(x, self.gt_boxes, self.gt_class_ids, self.gt_masks, '/home/yuruiqi/visualization')
        # observer.show_boxes_filt(channel=0, boxes=rpn_rois, score=self.vfm['rpn_scores'], threshold=0.5,
        #                          save_dir=r'/home/yuruiqi/visualization/rpn')

        # 5. Train or Inference or RPN
        if self.mode == 'RPN':
            return rpn_logits, rpn_scores, rpn_bboxes, None, None, None, None, None, None, None, anchors
        if self.mode == 'train':
            # TODO: Should reduce train_proposals_per_image to around 200. Set to 500 to avoid error when debugging.
            # DetectionTargetLayer
            rois, target_class_ids, target_bbox, target_mask = self.detection_target_layer.process(rpn_rois)

            # observer = Observer(x, self.gt_boxes, self.gt_class_ids, self.gt_masks, '/home/yuruiqi/visualization')
            # observer.show_boxes_filt(channel=0, boxes=rois, match=target_class_ids, match_score=2, save_dir=r'/home/yuruiqi/visualization/dtl_box')
            # refined_box = Utils.batch_slice([rois, target_bbox], Utils.refine_boxes)
            # observer.show_boxes_filt(channel=0, boxes=refined_box, match=target_class_ids, match_score=0, save_dir=r'/home/yuruiqi/visualization/dtl_box')

            # Network heads
            # (batch, n_rois, n_classes), (batch, n_rois, n_classes), (batch, n_rois, n_classes, 4)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(rois, mrcnn_feature_maps)

            # self.vfm.update(self.fpn_classifier.vfm)
            # Visualization.visualize_mask(self.vfm['fpn_classifier_roi_align'],
            #                              save_dir=r'/home/yuruiqi/visualization/classifier_roi_align/', n_watch=10, n_class_watch=10)

            # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
            mrcnn_mask = self.fpn_mask(rois, mrcnn_feature_maps)

            # mrcnn_mask.register_hook(grads.save_grad('mrcnn_mask'))
            # grads.print_grad('mrcnn_mask')

            # self.vfm.update(self.fpn_mask.vfm)
            # Visualization.visualize_mask(self.vfm['fpn_mask_roi_align'],
            #                              save_dir=r'/home/yuruiqi/visualization/mask_roi_align/', n_watch=10,
            #                              n_class_watch=10)

            return rpn_logits, rpn_scores, rpn_bboxes, \
                   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
                   target_class_ids, target_bbox, target_mask, anchors
        else:
            # Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(rpn_rois, mrcnn_feature_maps)

            # Detections
            # (batch, detection_max_instance, [y1, x1, y2, x2, class_id, score])
            detections = self.detection_layer.process(rpn_rois, mrcnn_class, mrcnn_bbox)
            detection_boxes = detections[:, :, 0:4]
            detection_classes = detections[:, :, 4]
            detection_scores = detections[:, :, 5]
            # (batch, detection_max_instance, n_classes, h_mask, w_mask)
            mrcnn_masks = self.fpn_mask(detection_boxes, mrcnn_feature_maps)

            # self.vfm.update(self.fpn_classifier.vfm)
            # self.vfm.update(self.fpn_mask.vfm)
            self.vfm.update(self.detection_layer.vfm)

            return detection_boxes, detection_classes, detection_scores, mrcnn_masks


def set_trainable(model, train_parts, train_bn=True):
    """
    train_part: list of 'Backbone' or 'RPN' or 'FPN_heads' or 'Heads'
    """
    if model.__class__.__name__ == 'DataParallel':
        model = model.module
    dict = {'Backbone': (model.resnet,), 'RPN': (model.fpn, model.rpn),
            'FPN_heads': (model.fpn, model.fpn_classifier, model.fpn_mask),
            'Heads': (model.fpn, model.rpn, model.fpn_classifier, model.fpn_mask)}

    if not train_bn:
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        model.apply(set_bn_eval)

    # Set all to not train
    for para in model.parameters():
        para.requires_grad = False

    for train_name in train_parts:
        part = dict[train_name]
        for net in part:
            for para in net.parameters():
                para.requires_grad = True
