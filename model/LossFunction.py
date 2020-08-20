import torch
import torch.nn as nn
from model.Utils import GradSaver, batch_slice, compute_overlaps, compute_deltas, trim_zero_graph
from model.Visualization import Observer
# from model import Utils
import numpy as np

grads = GradSaver()


###############
# Loss function
###############
class LossComputer:
    def __init__(self, loss_part, rpn_train_anchors_per_image=256):
        self.loss_part = loss_part
        self.loss_dict = {}

        self.rpn_train_anchors_per_image = rpn_train_anchors_per_image

        self.target_rpn_match = None
        self.target_rpn_bbox = None
        self.active_class_ids = None

    def get_loss(self, rpn_class_logits, rpn_bbox,
                pred_class_logits, pred_bbox, pred_masks,
                target_class_ids, target_bbox, target_masks):
        if self.loss_part in ['RPN', 'Heads']:
            self.compute_rpn_class_loss(rpn_class_logits)
            self.compute_rpn_bbox_loss(rpn_bbox)
        if self.loss_part in ['FPN_heads', 'Heads']:
            self.compute_mrcnn_class_loss(pred_class_logits, target_class_ids, self.active_class_ids)
            self.compute_mrcnn_bbox_loss(pred_bbox, target_bbox, target_class_ids)
            self.compute_mrcnn_mask_loss(pred_masks, target_masks, target_class_ids)
        loss = sum(self.loss_dict.values())
        loss_dict_item = {x: self.loss_dict[x].item() for x in self.loss_dict.keys()}
        return loss, loss_dict_item

    def compute_rpn_class_loss(self, rpn_class_logits):
        """RPN anchors classifier loss.

        rpn_class_logits: (batch, anchors, 2). RPN classifier logits for BG/FG.
        rpn_match: (batch, anchors, 1). Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
        """
        # Convert -1,0,1 to 0,1 .
        anchor_class = torch.where(self.target_rpn_match.eq(1), torch.tensor(1, device=self.target_rpn_match.device),
                                   torch.tensor(0, device=self.target_rpn_match.device))

        # Positive and Negative anchors contribute to the loss, but neutral anchors (match value = 0) don't.
        indices = torch.ne(self.target_rpn_match, 0)
        rpn_class_logits = rpn_class_logits[indices]
        anchor_class = anchor_class[indices]
        # TODO: rpn_class_logits can be 1 output?
        loss = nn.functional.cross_entropy(rpn_class_logits, anchor_class.to(torch.long),
                                           weight=torch.tensor([0.1, 0.9]).to(rpn_class_logits.device))

        self.loss_dict.update({'rpn_class':loss})

    def compute_rpn_bbox_loss(self, rpn_bbox):
        """
        RPN anchors bounding box loss.

        rpn_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)])
        target_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)])
        rpn_match: (batch, n_anchors)
        """
        # Positive anchors contribute to the loss, but negative and neutral anchors (match value of 0 or -1) don't.
        indices = torch.eq(self.target_rpn_match, 1)
        rpn_bbox = rpn_bbox[indices]
        target_bbox = self.target_rpn_bbox[indices]

        loss = nn.functional.smooth_l1_loss(rpn_bbox, target_bbox)
        self.loss_dict.update({'rpn_bbox':loss})

    def compute_mrcnn_class_loss(self, pred_class_logits, target_class_ids, active_class_ids):
        """
        Loss for the classifier head of Mask RCNN.
        TODO: different

        pred_class_logits: (batch, n_rois, num_classes)
                          Note: num_classes includes background.
        target_class_ids: (batch, n_rois). Integer class IDs. Uses zero padding to fill in the array.
        active_class_ids: (batch, num_classes). Has a value of 1 for classes that are in the dataset of the image, and 0
            for classes that are not in the dataset.
        """
        # 1. without activate
        pred_class_logits = pred_class_logits.reshape([-1, pred_class_logits.shape[2]])  # (batch*n_rois, num_classes)
        target_class_ids = target_class_ids.reshape([-1])  # (batch*n_rois)

        # show_ix = target_class_ids.gt(0)
        # print(target_class_ids[show_ix], pred_class_logits.argmax(dim=1)[show_ix])

        num_classes = pred_class_logits.shape[-1]
        weight = [1/(num_classes-1)]*num_classes
        weight[0] = 0.01
        weight = torch.tensor(weight, device=pred_class_logits.device)
        loss = nn.functional.cross_entropy(pred_class_logits, target_class_ids.to(torch.long), weight=weight)
        # loss = nn.functional.cross_entropy(pred_class_logits, target_class_ids.to(torch.long),
        #                                    weight=torch.tensor([0.01, 0.433, 0.433, 0.233]).to(pred_class_logits.device))
        # loss = nn.functional.cross_entropy(pred_class_logits, target_class_ids.to(torch.long),
        #                                    weight=torch.tensor([0.01, 0.499, 0.499]).to(pred_class_logits.device))


        # 2. with activate
        # pred_class_ids = torch.argmax(pred_class_logits, dim=2)  # (batch, n_rois)
        # # (batch, n_rois) to (batch*n_rois)
        # pred_active = batch_slice([active_class_ids, pred_class_ids],
        #                           lambda x,y: torch.index_select(x,index=y,dim=0)).reshape([-1]).to(torch.bool)
        # pred_class_logits = pred_class_logits.reshape([-1, pred_class_logits.shape[2]])  # (batch*n_rois, num_classes)
        # target_class_ids = target_class_ids.reshape([-1])  # (batch*n_rois)
        # if pred_active.nonzero().shape[0] != 0:
        #     # print(pred_class_logits[pred_active].argmax(dim=1), target_class_ids.to(torch.long)[pred_active])
        #     loss = nn.functional.cross_entropy(pred_class_logits[pred_active], target_class_ids.to(torch.long)[pred_active])
        # else:
        #     loss = torch.tensor(1.0, requires_grad=True, device=target_class_ids.device)
        #
        self.loss_dict.update({'mrcnn_class':loss})

    def compute_mrcnn_bbox_loss(self, pred_bbox, target_bbox, target_class_ids):
        """Loss for Mask R-CNN bounding box refinement.

        pred_bbox: (batch, n_rois, num_classes, [dy, dx, log(dh), log(dw)])
                   Note: num_classes includes background.
        target_bbox: (batch, n_rois, [dy, dx, log(dh), log(dw)])
        target_class_ids: (batch, n_rois). int.
        """
        pred_bbox = torch.reshape(pred_bbox, (-1, pred_bbox.shape[2], 4))  # (batch*n_rois, n_classes, 4)
        target_bbox = torch.reshape(target_bbox, (-1, 4))  # (batch*n_rois, 4)
        target_class_ids = torch.reshape(target_class_ids, (-1,))  # (batch*n_rois)

        # Only positive ROIs contribute to the loss. And only the right class_id of each ROI. Get their indices.
        positive_roi_ix = target_class_ids.gt(0).nonzero().squeeze(dim=1)  # (n_positive). roi_index
        # (n_positive). class_ids
        positive_roi_class_ids = torch.index_select(target_class_ids, dim=0, index=positive_roi_ix).to(torch.int64)
        indices = torch.stack([positive_roi_ix, positive_roi_class_ids], dim=1)  # (n_positive, [roi_index, class_ids])

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = torch.index_select(target_bbox, dim=0, index=positive_roi_ix)  # (n_positive, 4)
        pred_bbox = pred_bbox[indices[:, 0], indices[:, 1], :]  # (n_positive, 4)

        if target_bbox.shape[0] > 0:
            # loss = nn.functional.smooth_l1_loss(pred_bbox, target_bbox)
            loss = nn.functional.mse_loss(pred_bbox, target_bbox)
        else:
            loss = torch.tensor(1.0, requires_grad=True, device=pred_bbox.device)
        self.loss_dict.update({'mrcnn_bbox':loss})

    def compute_mrcnn_mask_loss(self, pred_masks, target_masks, target_class_ids):
        """Mask binary cross-entropy loss for the masks head.

        pred_masks: [batch, n_rois, num_classes, h_mask, w_mask] float32 tensor with values from 0 to 1.
                    Note: num_classes includes background.
        target_masks: (batch, n_rois, h_mask, w_mask). A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: (batch, n_rois). Int. Zero padded.
        """
        # Merge batch and n_rois dim
        pred_masks = pred_masks.reshape(
            (-1, pred_masks.shape[2], pred_masks.shape[3], pred_masks.shape[4]))  # (batch*n_rois,n_classes,h,w)
        target_masks = target_masks.reshape((-1, target_masks.shape[2], target_masks.shape[3]))  # (batch*n_rois,h,w)
        target_class_ids = target_class_ids.reshape((-1,))  # (batch*n_rois)

        # Only positive ROIs contribute to the loss. And only the right class_id of each ROI. Get their indices.
        positive_roi_ix = target_class_ids.gt(0).nonzero().squeeze(dim=1)  # (n_positive). roi_index
        # (n_positive). class_ids
        positive_roi_class_ids = torch.index_select(target_class_ids, dim=0, index=positive_roi_ix).to(torch.int64)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[positive_roi_ix]
        # TODO: class_id should -1 to be indice.
        y_pred = pred_masks[positive_roi_ix, positive_roi_class_ids]

        # y_pred.register_hook(grads.save_grad('y_pred_grad'))
        # grads.print_grad('y_pred_grad')

        # y_true = y_true.detach()
        if y_true.shape[0] > 0:
            loss = nn.functional.binary_cross_entropy(y_pred, y_true)
        else:
            loss = torch.tensor(1.0, requires_grad=True, device=y_pred.device)
        # TODO: Why mean?
        # loss = torch.mean(loss)
        self.loss_dict.update({'mrcnn_mask':loss})

    def build_rpn_targets(self, anchors, gt_boxes):
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
        gt_boxes, _ = trim_zero_graph(gt_boxes)

        # Set all to neutral first.
        rpn_match = torch.zeros([anchors.shape[0]], dtype=torch.int16)  # (n_anchors)
        rpn_bbox = torch.zeros((anchors.shape[0], 4))  # (n_anchors, [dy, dx, log(dh), log(dw)])

        overlaps = compute_overlaps(anchors, gt_boxes)  # (n_anchors, n_classes)

        # Match anchors to GT boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive(1).
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative(-1).
        # Neutral(0) anchors are those that don't match the conditions above, and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens).
        # Instead, match it to the closest anchor (even if its max IoU is < 0.3).

        # 1. Get matched IoU and index of the anchors. Set negative anchors.
        # TODO: Pytorch bug
        if overlaps.shape[1] == 1:
            anchor_iou_argmax = torch.zeros(overlaps.shape[0], dtype=torch.int64, device=overlaps.device)
        else:
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

        # Subsample to balance positive and negative training anchors.
        # 1. Don't let positives be more than half the training anchors.
        # TODO: 1.Why?     2. Need to use torch instead of np?
        ids = np.where(rpn_match == 1)[0]  # (n_positive)
        extra = len(ids) - self.rpn_train_anchors_per_image // 2
        if extra > 0:
            # Reset the extra ones to neutral randomly.
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        # 2. Don't let negatives be more than (training_anchors - positives).
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (self.rpn_train_anchors_per_image - torch.sum(rpn_match == 1)).cpu().detach().numpy()
        if extra > 0:
            # Rest the extra ones to neutral randomly.
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        #  First compute deltas to the corresponding GT boxes.
        not_positive_ids = torch.ne(rpn_match, 1)
        anchor_gt_boxes = torch.index_select(gt_boxes, dim=0, index=anchor_iou_argmax)
        rpn_bbox = compute_deltas(anchors, anchor_gt_boxes)
        # TODO: Must do this?
        # Then set not positive anchors' deltas to zero.
        rpn_bbox[not_positive_ids] = 0

        return rpn_match, rpn_bbox

    def get_rpn_targets(self, anchors, gt_boxes):
        """
        Get rpn targets.

        rpn_train_anchors_per_image: int. Num of anchors to get bbox. The left are zero padding.

        return:
            rpn_match: (batch, n_anchors). Matches between anchors and GT boxes.
                       1 = positive anchor, -1 = negative anchor, 0 = neutral
            rpn_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)]). Anchor bbox deltas.
        """
        # add batch dim
        batch_size = gt_boxes.shape[0]
        rpn_train_anchors_per_image = [self.rpn_train_anchors_per_image] * batch_size

        rpn_match, rpn_bbox = batch_slice([anchors, gt_boxes], self.build_rpn_targets)
        rpn_match = rpn_match.to(rpn_bbox.device)
        self.target_rpn_match = rpn_match
        self.target_rpn_bbox = rpn_bbox

    def get_active_class_ids(self, gt_class_ids, n_classes):
        """

        gt_class_ids: (batch, max_instances). From 1 to n_class. Have zero-padding.
        n_classes: int.

        return: (batch, num_classes)
                Note: num_classes includes background.
        """
        array = torch.zeros([gt_class_ids.shape[0], n_classes + 1], dtype=torch.int32, device=gt_class_ids.device)

        # trim zero-padding
        ix = gt_class_ids.gt(0).nonzero()
        ix_fill = torch.stack([ix[:, 0], (gt_class_ids[ix[:, 0], ix[:, 1]]).to(torch.int64)], dim=1)
        array[ix_fill[:, 0], ix_fill[:, 1]] = torch.tensor(1, dtype=torch.int32, device=gt_class_ids.device)

        self.active_class_ids = array

# TODO: nn.Module
