import torch
import torch.nn as nn

from model import Utils, Visualization
from model import LossFunction
from model.NetworkLayers import ResNet50, ResNetPyTorch, FPN, RPN, FPNClassifier, FPNMask
from model.FunctionLayers import ProposalLayer, DetectionTargetLayer, DetectionLayer

import numpy as np


# Mask R-CNN
class MRCNN(nn.Module):
    def __init__(self, in_channels, image_shape, mode):
        super().__init__()
        self.mode = mode
        self.image_shape = image_shape

        self.gt_class_ids = None
        self.gt_boxes = None
        self.gt_masks = None
        self.active_class_ids = None

        # visualization feature map
        self.vfm = {}

        # TODO: parameter "training" of batchnorm need to be set. (May be some batchnorm hasn't be written.)
        # self.resnet = ResNet50(in_channels, 2048)
        # TODO: channel
        self.resnet = ResNetPyTorch(pretrained=True)

        self.fpn = FPN(2048, 256)
        self.rpn = RPN(256)

        self.proposal_layer = ProposalLayer(post_nms_rois=300, nms_threshold=0.8, pre_nms_limit=1000)
        # self.detection_target_layer = DetectionTargetLayer(self.gt_class_ids, self.gt_boxes, self.gt_masks,
        #                                                    proposal_positive_ratio=0.33, train_proposals_per_image=100,
        #                                                    mask_shape=[28, 28])
        self.detection_layer = DetectionLayer(detection_max_instances=10, detection_nms_threshold=0.5)
        # in channel = fpn out channel, out channel = num_classes
        self.fpn_classifier = FPNClassifier(256, 3, fc_layers_size=1024, pool_size=7, image_shape=image_shape)
        self.fpn_mask = FPNMask(256, 3, mask_pool_size=14, image_shape=image_shape)

        # Get when forwarding.
        self.anchors = None  # (batch, n_anchors, 4). Same among batch so don't need batch dim.

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

        # pyramid shape is [batch, channels, h, w]. So take last 2 as shape.
        pyramid_shapes = [[x.shape[-2], x.shape[-1]] for x in rpn_feature_maps]
        # TODO: It should be confirmed.
        feature_strides = [4, 8, 16, 32, 64]
        # generate anchors
        self.generate_anchors(pyramid_shapes, feature_strides, p2.shape[0])

        # 4. Proposal Layer
        rpn_rois = self.proposal_layer.process(self.anchors, rpn_scores, rpn_bboxes)
        self.vfm['rpn_rois'] = rpn_rois

        # 5. Train or Inference
        if self.mode == 'train':
            # TODO: Should reduce train_proposals_per_image to around 200. Set to 500 to avoid error when debugging.
            # DetectionTargetLayer
            rois, target_class_ids, target_bbox, target_mask = self.detection_target_layer.process(rpn_rois)
            # visualize_box = Utils.batch_slice([rois, target_bbox], Utils.refine_boxes)
            # Visualization.visualize_boxes(x, visualize_box, target_class_ids,
            #                               save_dir='/home/yuruiqi/visualization/dtl_rois/', view_batch=0)
            # Visualization.visualize_mask(target_mask.unsqueeze(dim=2), save_dir='/home/yuruiqi/visualization/dtl_mask/')

            # Network heads
            # (batch, n_rois, n_classes), (batch, n_rois, n_classes), (batch, n_rois, n_classes, 4)
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier(rois, mrcnn_feature_maps)
            # self.vfm.update(self.fpn_classifier.vfm)
            # Visualization.visualize_mask(self.vfm['fpn_classifier_roi_align'],
            #                              save_dir=r'/home/yuruiqi/visualization/classifier_roi_align/', n_watch=10, n_class_watch=10)

            # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
            mrcnn_mask = self.fpn_mask(rois, mrcnn_feature_maps)
            # self.vfm.update(self.fpn_mask.vfm)
            # Visualization.visualize_mask(self.vfm['fpn_mask_roi_align'],
            #                              save_dir=r'/home/yuruiqi/visualization/mask_roi_align/', n_watch=10,
            #                              n_class_watch=10)

            return rpn_logits, rpn_scores, rpn_bboxes, \
                   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
                   target_class_ids, target_bbox, target_mask
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

            return detection_boxes, detection_classes, detection_scores, mrcnn_masks

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
        anchors = torch.tensor(anchors, dtype=torch.float32).cuda()
        self.anchors = anchors

    def build_rpn_targets(self, anchors, gt_boxes, rpn_train_anchors_per_image):
        """Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.
        Note that only the anchor whose match is 1 or -1 has bbox. Others are zero paddings.

        anchors: (n_anchors, [y1, x1, y2, x2])
        gt_class_ids: (n_classes)
        gt_boxes: (n_classes, [y1, x1, y2, x2])

        TODO: Note that it's written mistakenly in the original code.
        rpn_match: (n_anchors). Matches between anchors and GT boxes. 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: (n_anchors, [dy, dx, log(dh), log(dw)]). Anchor bbox deltas.
        """
        # Added by myself. Eliminate zero padding gt boxes. So rpn_targets will not regress to there.
        gt_boxes, _ = Utils.trim_zero_graph(gt_boxes)

        # Set all to neutral first.
        rpn_match = torch.zeros([anchors.shape[0]], dtype=torch.int32)  # (n_anchors)
        rpn_bbox = torch.zeros((anchors.shape[0], 4))  # (n_anchors, [dy, dx, log(dh), log(dw)])

        overlaps = Utils.compute_overlaps(anchors, gt_boxes)  # (n_anchors, n_classes)

        # Match anchors to GT boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive(1).
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative(-1).
        # Neutral(0) anchors are those that don't match the conditions above, and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens).
        # Instead, match it to the closest anchor (even if its max IoU is < 0.3).

        # 1. Get matched IoU and index of the anchors. Set negative anchors.
        anchor_iou_argmax = torch.argmax(overlaps, dim=1)  # (n_anchors), best matched gt_boxes of every anchor.
        anchor_iou_max = torch.max(overlaps, dim=1)[0]  # (n_anchors), best matched overlaps of every anchor.
        rpn_match[anchor_iou_max < 0.3] = -1
        # 2. Set an anchor for each GT box (regardless of IoU value).
        # If multiple anchors have the same IoU, then match all of them.
        # index of all best anchors to gt boxes.
        gt_iou_argmax = torch.nonzero(overlaps == torch.max(overlaps, dim=0)[0])[:, 0]
        rpn_match[gt_iou_argmax] = 1
        # 3. Set anchors with high overlap as positive. TODO: 0.7 maybe to high???
        rpn_match[anchor_iou_max >= 0.7] = 1

        # Subsample to balance positive and negative anchors.
        # 1. Don't let positives be more than half the anchors.
        # TODO: 1.Why?     2. Need to use torch instead of np?
        ids = np.where(rpn_match == 1)[0]  # (n_positive)
        extra = len(ids) - rpn_train_anchors_per_image // 2
        if extra > 0:
            # Reset the extra ones to neutral randomly.
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        # 2. Don't let negatives be more than (anchors - positives).
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (rpn_train_anchors_per_image - torch.sum(rpn_match == 1)).cpu().detach().numpy()
        if extra > 0:
            # Rest the extra ones to neutral randomly.
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        #  First compute deltas to the corresponding GT boxes.
        not_positive_ids = torch.ne(rpn_match, 1)
        anchor_gt_boxes = torch.index_select(gt_boxes, dim=0, index=anchor_iou_argmax)
        rpn_bbox = Utils.compute_deltas(anchors, anchor_gt_boxes)
        # TODO: Must do this?
        # Then set not positive anchors' deltas to zero.
        rpn_bbox[not_positive_ids] = 0

        return rpn_match, rpn_bbox

    def get_rpn_targets(self, rpn_train_anchors_per_image=256):
        """
        Get rpn targets.

        rpn_train_anchors_per_image: int. Num of anchors to get bbox. The left are zero padding.

        return:
            rpn_match: (batch, n_anchors). Matches between anchors and GT boxes.
                       1 = positive anchor, -1 = negative anchor, 0 = neutral
            rpn_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)]). Anchor bbox deltas.
        """
        # add batch dim
        rpn_train_anchors_per_image = [rpn_train_anchors_per_image] * self.anchors.shape[0]
        rpn_match, rpn_bbox = Utils.batch_slice(
            [self.anchors, self.gt_boxes, rpn_train_anchors_per_image], self.build_rpn_targets)

        rpn_match = rpn_match.cuda()
        return rpn_match, rpn_bbox

    def set_train_gt(self, gt_class_ids, gt_boxes, gt_masks):
        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks
        self.active_class_ids = torch.where(self.gt_class_ids == 0,
                                            torch.tensor(0, dtype=torch.int32).cuda(),
                                            torch.tensor(1, dtype=torch.int32).cuda())

        self.detection_target_layer = DetectionTargetLayer(gt_class_ids, gt_boxes, gt_masks,
                                                           proposal_positive_ratio=0.33, train_proposals_per_image=100,
                                                           mask_shape=[28, 28])

    def set_trainable(self, train_parts):
        """
        train_part: list of 'Backbone' or 'RPN' or 'MRCNN'
        """
        # TODO: confirm
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(set_bn_eval)

        dict = {'Backbone':(self.resnet,), 'RPN':(self.fpn, self.rpn),
                'MRCNN':(self.fpn, self.fpn_classifier, self.fpn_mask)}

        # Set all to not train
        for para in self.parameters():
            para.requires_grad = False

        for train_name in train_parts:
            part = dict[train_name]
            for net in part:
                for para in net.parameters():
                    para.requires_grad = True

    def train_part(self, images, gt_class_ids, gt_boxes, gt_masks, part=None):
        # TODO: Confirm the relation between require_grad and optim
        self.set_train_gt(gt_class_ids, gt_boxes, gt_masks)

        # get pred and label
        rpn_logits, rpn_scores, rpn_bboxes, \
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        target_class_ids, target_bbox, target_mask = self(images)

        target_rpn_match, target_rpn_bbox = self.get_rpn_targets()

        # visualize amchors and target boxes and masks
        # Visualization.visualize_boxes(images, self.anchors, save_dir='/home/yuruiqi/visualization/anchors/',
        #                               view_batch=0, n_watch=200)
        # Visualization.visualize_rpn_targets(images, self.anchors, target_rpn_bbox, target_rpn_match,
        #                                     save_dir='/home/yuruiqi/visualization/target_rpn_box/', n_watch=50)
        # Visualization.visualize_target_mask(target_mask, save_dir='/home/yuruiqi/visualization/target_mask/')

        # Compute Loss
        if part == 'RPN':
            loss, loss_dict = LossFunction.compute_rpn_loss(rpn_logits, rpn_bboxes,
                                                            target_rpn_match, target_rpn_bbox)
        elif part == 'MRCNN':
            loss, loss_dict = LossFunction.compute_head_loss(mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                                                             target_class_ids, target_bbox, target_mask,
                                                             self.active_class_ids)
        elif part == 'Heads':
            loss, loss_dict = LossFunction.compute_loss(rpn_logits, rpn_bboxes, target_rpn_match, target_rpn_bbox,
                                                        mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                                                        target_class_ids, target_bbox, target_mask,
                                                        self.active_class_ids)
        else:
            raise ValueError("Rua")
        return loss, loss_dict

    def load_weight(self, path):
        self.resnet.load_state_dict(torch.load(path))
