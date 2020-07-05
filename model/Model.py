import torch
import torch.nn as nn
from model import Utils, LossFunction, Visualization
import numpy as np

from model.NetworkLayers import ResNet, FPN, RPN, FPNClassifier, FPNMask
from model.FunctionLayers import ProposalLayer, DetectionTargetLayer, DetectionLayer
from dataset import DataLoader


# Mask R-CNN
class MRCNN(nn.Module):
    def __init__(self, in_channels, image_shape, mode, gt_class_ids=None, gt_boxes=None, gt_masks=None):
        super().__init__()
        self.mode = mode
        self.image_shape = image_shape

        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks
        if mode == 'train':
            self.active_class_ids = torch.where(gt_class_ids == 0,
                                                torch.tensor(0, dtype=torch.int32).cuda(),
                                                torch.tensor(1, dtype=torch.int32).cuda())

        # visualization feature map
        self.vfm = {}

        # TODO: parameter "training" of batchnorm need to be set. (May be some batchnorm hasn't be written.)
        self.resnet = ResNet(in_channels, 2048)
        self.fpn = FPN(2048, 256)
        self.rpn = RPN(256)

        self.proposal_layer = ProposalLayer(post_nms_rois=300, nms_threshold=0.8, pre_nms_limit=1000)
        self.detection_target_layer = DetectionTargetLayer(gt_class_ids, gt_boxes, gt_masks,
                                                           proposal_positive_ratio=0.33, train_proposals_per_image=100,
                                                           mask_shape=[28, 28])
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

        self.vfm["rpn_feature_maps"] = rpn_feature_maps

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
        anchors = self.generate_anchors(pyramid_shapes, feature_strides, p2.shape[0]).cuda()

        # 4. Proposal Layer
        rpn_rois = self.proposal_layer.process(anchors, rpn_scores, rpn_bboxes)
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
            self.vfm.update(self.fpn_classifier.vfm)
            # Visualization.visualize_mask(self.vfm['fpn_classifier_roi_align'],
            #                              save_dir=r'/home/yuruiqi/visualization/classifier_roi_align/', n_watch=10, n_class_watch=10)

            # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
            mrcnn_mask = self.fpn_mask(rois, mrcnn_feature_maps)
            self.vfm.update(self.fpn_mask.vfm)
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

            self.vfm.update(self.fpn_classifier.vfm)
            self.vfm.update(self.fpn_mask.vfm)

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
        anchors = torch.tensor(anchors, dtype=torch.float32)
        self.anchors = anchors.cuda()
        return anchors

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
            [self.anchors, self.gt_boxes, rpn_train_anchors_per_image], DataLoader.build_rpn_targets)

        return rpn_match, rpn_bbox

    def set_trainable(self, train_part='ALL'):
        """
        train_part: str. 'RPN' or 'Head' or 'ALL'
        """
        rpn_part = [self.resnet, self.fpn, self.rpn]
        head_part = [self.resnet, self.fpn, self.fpn_classifier, self.fpn_mask]
        all_part = rpn_part + head_part

        # Only RPN part train
        if train_part == 'RPN':
            for net in rpn_part:
                for para in net.parameters():
                    para.requires_grad = True
            for net in head_part:
                for para in net.parameters():
                    para.requires_grad = False

        # Only Head part train
        if train_part == 'Head':
            for net in rpn_part:
                for para in net.parameters():
                    para.requires_grad = False
            for net in head_part:
                for para in net.parameters():
                    para.requires_grad = True

        # All part train
        if train_part == 'All':
            for net in all_part:
                for para in net.parameters():
                    para.requires_grad = True

    def train_part(self, images, save_path, part, lr, epoch):
        # TODO: Confirm the relation between require_grad and optim
        self.set_trainable(part)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        min_loss = None
        for i in range(epoch):
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
            elif part == 'Head':
                loss, loss_dict = LossFunction.compute_head_loss(mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                                                                 target_class_ids, target_bbox, target_mask,
                                                                 self.active_class_ids)
            elif part == 'All':
                loss, loss_dict = LossFunction.compute_loss(rpn_logits, rpn_bboxes, target_rpn_match, target_rpn_bbox,
                                                            mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                                                            target_class_ids, target_bbox, target_mask,
                                                            self.active_class_ids)
            else:
                raise("My dear, 'part' should be 'RPN' or 'Head' or 'All'.")

            # save model
            print(part, loss.item())
            print(loss_dict)
            patience = 0
            if (not min_loss) or loss < min_loss:
                print('save')
                min_loss = loss
                torch.save(self.state_dict(), save_path)
                patience = 0
            else:
                patience += 1
            print('')

            # patience break
            if patience == 10:
                break

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


