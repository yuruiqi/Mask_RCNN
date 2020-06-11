import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Utils
from torch.autograd import Variable


def build_rpn_targets(anchors, gt_boxes, rpn_train_anchors_per_image):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: (n_anchors, [y1, x1, y2, x2])
    gt_class_ids: (n_classes)
    gt_boxes: (n_classes, [y1, x1, y2, x2])

    TODO: Note that it's written mistakenly is the original code.
    rpn_match: (n_anchors). Matches between anchors and GT boxes. 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: (n_anchors, [dy, dx, log(dh), log(dw)]). Anchor bbox deltas.
    """
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
    anchor_iou_argmax = torch.argmax(overlaps, dim=1)  # (n_anchors), best matched gt_boxes.
    anchor_iou_max = torch.max(overlaps, dim=1)[0]  # (n_anchors), best matched overlaps.
    rpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU, then match all of them.
    # index of all best anchors to gt boxes.
    gt_iou_argmax = torch.nonzero(overlaps == torch.max(overlaps, dim=0)[0])[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors.
    # 1. Don't let positives be more than half the anchors.
    # TODO: 1.Why?     2. Need to use torch instead of np?
    ids = np.where(rpn_match == 1)[0]  # (n_positive)
    extra = len(ids) - rpn_train_anchors_per_image//2
    if extra > 0:
        # Reset the extra ones to neutral randomly.
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # 2. Don't let negatives be more than (anchors - positives).
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (rpn_train_anchors_per_image - torch.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral randomly.
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute deltas to the corresponding GT boxes.
    not_positive_ids = torch.ne(rpn_match, 1)
    anchor_gt_boxes = torch.index_select(gt_boxes, dim=0, index=anchor_iou_argmax)
    rpn_bbox = Utils.compute_deltas(anchors, anchor_gt_boxes)
    # TODO: Must do this?
    # Set not positive anchors' deltas to zero.
    rpn_bbox[not_positive_ids] = 0

    return rpn_match, rpn_bbox


if __name__ == '__main__':
    from artifical_data import _GenerateCircle
    image, classes, boxes, masks = _GenerateCircle()

    image = torch.tensor(image)
    classes = torch.tensor(classes)  # (n_classes)
    boxes = torch.tensor(boxes, dtype=torch.float32)  # (n_classes, 4)
    masks = torch.tensor(masks)

    pyramid_shapes = [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
    feature_strides = [4, 8, 16, 32, 64]
    anchor_genertor = Utils.AnchorGenerator(scales=[32, 64, 96], ratios=[0.5, 1, 1])
    anchors = anchor_genertor.get_anchors(pyramid_shapes, feature_strides)
    anchors = torch.tensor(anchors, dtype=torch.float32)  # (n_anchors, 4)

    rpn_train_anchors_per_image = [256] * anchors.shape[0]
    build_rpn_targets(anchors, boxes[0], rpn_train_anchors_per_image[0])
