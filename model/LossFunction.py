import torch
import torch.nn as nn
from model import Utils, Visualization


def compute_rpn_class_loss(rpn_class_logits, rpn_match):
    """RPN anchors classifier loss.

    rpn_class_logits: (batch, anchors, 2). RPN classifier logits for BG/FG.
    rpn_match: (batch, anchors, 1). Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    """
    # Convert -1,0,1 to 0,1 .
    rpn_match = torch.where(rpn_match==1, torch.tensor(1), torch.tensor(0))

    # Positive and Negative anchors contribute to the loss, but neutral anchors (match value = 0) don't.
    indices = torch.ne(rpn_match, 0)
    rpn_class_logits = rpn_class_logits[indices]
    rpn_match = rpn_match[indices].cuda()

    loss = nn.functional.cross_entropy(rpn_class_logits, rpn_match.to(torch.long))
    return loss


def compute_rpn_bbox_loss(rpn_bbox, target_bbox, rpn_match):
    """
    RPN anchors bounding box loss.

    rpn_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)])
    target_bbox: (batch, n_anchors, [dy, dx, log(dh), log(dw)])
    rpn_match: (batch, n_anchors)
    """
    # Positive anchors contribute to the loss, but negative and neutral anchors (match value of 0 or -1) don't.
    indices = torch.eq(rpn_match, 1)
    rpn_bbox = rpn_bbox[indices]
    target_bbox = target_bbox[indices]

    loss = nn.functional.smooth_l1_loss(rpn_bbox, target_bbox)
    return loss


def compute_mrcnn_class_loss(pred_class_logits, target_class_ids, active_class_ids):
    """
    Loss for the classifier head of Mask RCNN.

    pred_class_logits: (batch, n_rois, n_classes)
    target_class_ids: (batch, n_rois). Integer class IDs. Uses zero padding to fill in the array.
    active_class_ids: (batch, n_classes). Has a value of 1 for classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # Note that here class ids is from zero (corresponding to dim).
    pred_class_ids = torch.argmax(pred_class_logits, dim=2)  # (batch, n_rois). from 0 to n_classes-1
    pred_active = Utils.batch_slice([active_class_ids, pred_class_ids],
                                    lambda x, y: torch.index_select(x, dim=0, index=y))  # (batch, n_rois). 0 and 1
    target_class_ids = Utils.batch_slice([target_class_ids, pred_active], torch.mul)

    # TODO: Need to confirm. Maybe merge batch and roi dimensions?
    # Different: target_class_ids is in [1, n_classes] with zero padding. So need to minus 1.
    # Than zero padding will be -1, so trim it first.
    target_class_ids_from0 = target_class_ids-1
    ix = target_class_ids_from0.gt(-1)
    pred_class_logits = pred_class_logits[ix]
    target_class_ids_from0 = target_class_ids_from0[ix]
    loss = nn.functional.cross_entropy(pred_class_logits, target_class_ids_from0.to(torch.long))

    # TODO: Former code
    # loss = Utils.batch_slice([pred_class_logits, target_class_ids.to(torch.long) - 1],
    #                          lambda x, y: nn.functional.cross_entropy(x, y))  # (batch)
    # Erase losses of predictions of classes that are not in the active classes of the image.
    # loss = Utils.batch_slice([loss, pred_active], torch.mul)  # (batch, n_rois)

    # Computer loss mean. Use only predictions that contribute to the loss to get a correct mean.
    # loss = torch.sum(loss) / torch.sum(pred_active)
    return loss


def compute_mrcnn_bbox_loss(pred_bbox, target_bbox, target_class_ids):
    """Loss for Mask R-CNN bounding box refinement.

    pred_bbox: (batch, n_rois, n_classes, [dy, dx, log(dh), log(dw)])
    target_bbox: (batch, n_rois, [dy, dx, log(dh), log(dw)])
    target_class_ids: (batch, n_rois). int.
    """
    pred_bbox = torch.reshape(pred_bbox, (-1, pred_bbox.shape[2], 4))  # (batch*n_rois, n_classes, 4)
    target_bbox = torch.reshape(target_bbox, (-1, 4))  # (batch*n_rois, 4)
    target_class_ids = torch.reshape(target_class_ids, (-1,))  # (batch*n_rois)

    # Only positive ROIs contribute to the loss. And only the right class_id of each ROI. Get their indices.
    positive_roi_ix = torch.nonzero(torch.gt(target_class_ids, 0)).squeeze(dim=1)  # (n_positive). roi_index
    # (n_positive). class_ids
    positive_roi_class_ids = torch.index_select(target_class_ids, dim=0, index=positive_roi_ix).to(torch.int64)
    # TODO: should minus 1 to be indice.
    indices = torch.stack([positive_roi_ix, positive_roi_class_ids-1], dim=1)  # (n_positive, [roi_index, class_ids])

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = torch.index_select(target_bbox, dim=0, index=positive_roi_ix)  # (n_positive, 4)
    pred_bbox = pred_bbox[indices[:, 0], indices[:, 1], :]  # (n_positive, 4)

    if target_bbox.shape[0] > 0:
        loss = nn.functional.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.tensor(0.0)

    # TODO: Why mean?
    loss = torch.mean(loss)
    return loss


def compute_mrcnn_mask_loss(pred_masks, target_masks, target_class_ids):
    """Mask binary cross-entropy loss for the masks head.

    pred_masks: [batch, n_rois, n_classes, h_mask, w_mask] float32 tensor with values from 0 to 1.
    target_masks: (batch, n_rois, h_mask, w_mask). A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: (batch, n_rois). Int. Zero padded.
    """
    # Merge batch and n_rois dim
    pred_masks = pred_masks.reshape((-1, pred_masks.shape[2], pred_masks.shape[3], pred_masks.shape[4]))  # (batch*n_rois,n_classes,h,w)
    target_masks = target_masks.reshape((-1, target_masks.shape[2], target_masks.shape[3]))  # (batch*n_rois,h,w)
    target_class_ids = target_class_ids.reshape((-1,))  # (batch*n_rois)

    # Only positive ROIs contribute to the loss. And only the right class_id of each ROI. Get their indices.
    positive_roi_ix = target_class_ids.gt(0).nonzero().squeeze(dim=1)  # (n_positive). roi_index
    # (n_positive). class_ids
    positive_roi_class_ids = torch.index_select(target_class_ids, dim=0, index=positive_roi_ix).to(torch.int64)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = target_masks[positive_roi_ix]
    # TODO: class_id should -1 to be indice.
    y_pred = pred_masks[positive_roi_ix, positive_roi_class_ids-1]

    if y_true.shape[0] > 0:
        loss = nn.functional.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.tensor(0.0)
    # TODO: Why mean?
    # loss = torch.mean(loss)
    return loss


def compute_rpn_loss(rpn_class_logits, rpn_bbox, rpn_match, target_rpn_bbox):
    rpn_class_loss = compute_rpn_class_loss(rpn_class_logits, rpn_match)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, target_rpn_bbox, rpn_match)
    loss = rpn_class_loss + rpn_bbox_loss

    loss_dict = {'rpn_class_loss': rpn_class_loss.item(), 'rpn_bbox_loss':rpn_bbox_loss.item()}
    return loss, loss_dict


def compute_head_loss(pred_class_logits, pred_bbox, pred_masks,
                      target_class_ids, target_bbox, target_masks, active_class_ids):
    mrcnn_class_loss = compute_mrcnn_class_loss(pred_class_logits, target_class_ids, active_class_ids)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(pred_bbox, target_bbox, target_class_ids)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(pred_masks, target_masks, target_class_ids)
    loss = mrcnn_class_loss + mrcnn_bbox_loss + 1000*mrcnn_mask_loss

    loss_dict = {'mrcnn_class_loss': mrcnn_class_loss.item(), 'mrcnn_bbox_loss': mrcnn_bbox_loss.item(),
                 'mrcnn_mask_loss': mrcnn_mask_loss.item()}
    return loss, loss_dict


def compute_loss(rpn_class_logits, rpn_bbox,
                rpn_match, target_rpn_bbox,
                pred_class_logits, pred_bbox, pred_masks,
                target_class_ids, target_bbox, target_masks,
                active_class_ids):
    rpn_loss, rpn_loss_dict = compute_rpn_loss(rpn_class_logits, rpn_bbox, rpn_match, target_rpn_bbox)
    head_loss, head_loss_dict = compute_head_loss(pred_class_logits, pred_bbox, pred_masks,
                                                  target_class_ids, target_bbox, target_masks, active_class_ids)
    loss = rpn_loss + head_loss

    loss_dict = rpn_loss_dict.copy()
    loss_dict.update(head_loss_dict)
    return loss, loss_dict
