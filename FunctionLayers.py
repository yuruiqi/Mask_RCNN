import numpy as np
import tensorflow as tf
import torch
from torchvision import ops
import Utils
from roi_align import RoIAlign
import math
import numpy as np


# Proposal Layer
class ProposalLayer:
    """
    Receives anchor scores and selects a subset to pass as proposals to the second stage. Filtering is done based on
    anchor scores and non-max suppression to remove overlaps. It also applies bounding box refinement deltas to anchors.

    The num_anchors is calculated by sigma(Pn_H * Pn_W * anchors_per_pixel).
    The anchors is arranged in order of anchors, transverse, longitude.
    """

    def __init__(self, proposal_count, nms_threshold=0.5, pre_nms_limit=6000):
        """
        proposal_count: ROIs kept after non-maximum suppression
        nms_threshold:  Threshold of IOU to perform nms. Default to be 0.5.
        pre_nms_limt: ROIs kept after tf.nn.top_k and before non-maximum suppression. Default to be 6000.
        """
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.pre_nms_limit = pre_nms_limit

    def process(self, anchors, scores, deltas):
        """
        anchors: (batch, num_anchors, [y1, x1, y2, x2]) anchors in normalized coordinates
        scores: (batch, num_anchors, [bg prob, fg prob])
        deltas: (batch, num_anchors, [dy, dx, log(dh), log(dw)])
        """
        # [batch, num_anchors, fg_prob]
        scores = scores[:, :, 1]
        # TODO: Bounding box refinement standard deviation on deltas?

        # Filter out top N(pre_nms_limit) rois according to the scores and get their indices.
        scores, ix = torch.topk(scores, k=self.pre_nms_limit, dim=-1, sorted=True)
        deltas = Utils.batch_slice([deltas, ix], lambda x, y: torch.index_select(x, dim=0, index=y))
        anchors = Utils.batch_slice([anchors, ix], lambda x, y: torch.index_select(x, dim=0, index=y))

        # Apply deltas to anchors to get refined boxes. [batch, N, (y1, x1, y2, x2)]
        boxes = Utils.batch_slice([anchors, deltas], lambda x, y: Utils.refine_boxes(x, y))

        # Clip boxes
        window = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        boxes = Utils.batch_slice(boxes, lambda x: Utils.clip_boxes(x, window))

        # nms
        proposals = Utils.batch_slice([boxes, scores], self.nms)
        return proposals

    def nms(self, boxes, scores):
        """
        Operate non_maximal_suppresion on boxes.
        boxes: (N, [y1, x1, y2, x2])
        scores: (N, [fg_probs])

        return:  Remained boxes after nms.
        """
        # TODO: Because torch can't limit the proposal_count, maybe it should be realized by myself.
        # if there is more box than the proposal_count, then nms is omitted.
        if boxes.shape[0] > self.proposal_count:
            indices = ops.nms(boxes, scores, self.nms_threshold)
            proposals = torch.index_select(boxes, dim=0, index=indices)
        else:
            proposals = boxes

        # Pad the batch slice so that it can be concatenated again.
        padding_count = max(self.proposal_count - proposals.shape[0], 0)
        # This parameter "pad" means filling on dim=-2 and in the bottom.
        proposals = torch.nn.functional.pad(proposals, pad=[0, 0, 0, padding_count])
        return proposals


# transform roi coordinates
def transform_coordianates(rois, image_shape):
    """
    Function RoIAlign is different from tf.crop_and_resize.
    ROI isn't in normalized coordinates and in (x1, y1, x2, y2) form.
    So transform roi coordinates from normalized (y1, x1, y2, x2) to unnormalized(x1, y1, x2, y2).

    rois: (N, (y1, x1, y2, x2))
    image_shape: [h, w]

    return: (N, (x1, y1, x2, y2))
    """
    y1, x1, y2, x2 = torch.split(rois, 1, dim=1)
    h, w = image_shape

    y1 = y1 * h
    x1 = x1 * w
    y2 = y2 * h
    x2 = x2 * w

    results = torch.cat([x1,y1,x2,y2], dim=1)
    return results


# Detection Target Layer
class DetectionTargetLayer:
    """
    Subsamples proposals by splitting positive and negative proposals.
    """
    def __init__(self, gt_class_ids, gt_boxes, gt_masks, proposal_positive_ratio, train_proposals_per_image,
                 mask_shape):
        """
        gt_class_ids: ()
        gt_boxes: (batch, MAX_GT_INSTANCES, [y1, x1, y2, x2]) in normalized coordinates.
        gt_masks: (batch, MAX_GT_INSTANCES, height, width) of boolean type
        proposal_positive_ratio: float. Percent of positive ROIs in all rois used to train classifier/mask heads.
        train_proposals_per_image: int. Number of ROIs per image to feed to classifier/mask heads
        mask_shape: [h, w]. Shape of output mask
        """
        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks
        self.roi_positive_ratio = proposal_positive_ratio
        self.train_rois_per_image = train_proposals_per_image
        self.mask_shape = mask_shape

    def process(self, proposals):
        outputs = Utils.batch_slice([proposals, self.gt_class_ids, self.gt_boxes, self.gt_masks],
                                    self.detection_targets_graph)
        return outputs

    def detection_targets_graph(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """
        Subsample proposals for one image (i.e. one batch) by splitting positive and negative proposals.

        proposals: (N, [y1, x1, y2, x2]). Proposals in normalized coordinates after ProposalLayer. Might be zero padded
                   if there are not enough  proposals.
        gt_class_ids: (all_GT_instances). Class IDs.
        gt_boxes: (all_GT_instances, [y1, x1, y2, x2]). Ground truth boxes in normalized coordinates.
        gt_masks: (all_GT_instances, height, width). Ground truth masks of boolen type.

        return:
            rois: (n, 4)
            roi_gt_class_ids: (n)
            deltas: (n, 4)
            roi_gt_masks_minibox: (n, mask_h, mask_w)
        """
        # Remove zero padding
        proposals, _ = Utils.trim_zero_graph(proposals)
        gt_boxes, non_zeros_ix = Utils.trim_zero_graph(gt_boxes)
        gt_class_ids = torch.index_select(gt_class_ids, dim=0, index=non_zeros_ix)
        gt_masks = torch.index_select(gt_masks, dim=0, index=non_zeros_ix)

        # Compute overlaps. overlaps (n_proposals, n_gt_boxes)
        overlaps = Utils.overlaps_graph(proposals, gt_boxes)

        # Determine positive and negative ROIs.
        # Get the max IoU of the proposal to all gt boxes.
        proposal_iou_max, _ = torch.max(overlaps, dim=1)
        # Positive rois are those with >= 0.5 IoU with a GT box.
        # Positive rois are those with < 0.5 IoU with every GT box.
        positive_roi_bool = torch.gt(proposal_iou_max, torch.tensor([0.5], dtype=torch.float32))
        positive_ix = torch.nonzero(positive_roi_bool)
        negative_ix = torch.nonzero(~positive_roi_bool)

        # Subsample rois to make positive/all = proposal_positive_ratio
        # 1. Positive rois (proposal_positive_ratio * train_proposals_per_image, 4)
        positive_count = int(self.roi_positive_ratio * self.train_rois_per_image)
        # TODO: Need shuffle on positive_ix and negative_ix before index selecting.
        positive_ix = positive_ix[0:positive_count].squeeze(1)
        # 2. Negative rois ((1-1/proposal_positive_ratio)*positive_count, 4)
        # Should calculated by this formula because positive rois may br not enough.
        negative_count = int((1 - 1 / self.roi_positive_ratio) * positive_count)
        negative_ix = negative_ix[0:negative_count].squeeze(1)
        # 3. Gather selected rois
        positive_rois = torch.index_select(proposals, dim=0, index=positive_ix)
        negative_rois = torch.index_select(proposals, dim=0, index=negative_ix)

        # Assign positive rois to corresponding GT boxes
        # positive overlaps: (n_positive, n_gt_boxes)
        positive_overlaps = torch.index_select(overlaps, dim=0, index=positive_ix)
        # roi_gt_box_assignment: (n_positive), best corresponding GT box ids of every ROI
        if positive_overlaps.shape[0] > 0:
            roi_gt_box_assignment = torch.argmax(positive_overlaps, dim=1)
        else:
            roi_gt_box_assignment = torch.tensor([], dtype=torch.int64)
        # roi_gt_boxes: (n_positive, 4). roi_gt_class_ids: (n_positive)
        roi_gt_boxes = torch.index_select(gt_boxes, dim=0, index=roi_gt_box_assignment)
        roi_gt_class_ids = torch.index_select(gt_class_ids, dim=0, index=roi_gt_box_assignment)

        # Compute deltas from positive_rois to roi_gt_boxes. (n_positive, 4)
        # TODO: BBOX_STD_DEV?
        deltas = Utils.compute_deltas(positive_rois, roi_gt_boxes)

        # Assign positive ROIs to corresponding GT masks. And permute to (n_positive, 1, height, weight)
        permuted_gt_masks = torch.unsqueeze(gt_masks, dim=1)
        roi_gt_masks = torch.index_select(permuted_gt_masks, dim=0, index=roi_gt_box_assignment)

        # Get masks in roi boxes. (n_positive, mask_h, mask_w)
        # TODO: normalize_to_mini_mask?
        positive_rois_transformed = transform_coordianates(positive_rois, gt_masks.shape[1:])
        box_ids = torch.arange(0, roi_gt_masks.shape[0], dtype=torch.int32)
        roi_align = RoIAlign(self.mask_shape[0], self.mask_shape[1])
        roi_gt_masks_minibox = roi_align(roi_gt_masks, positive_rois_transformed, box_ids)
        # Remove the extra dimension from masks.
        roi_gt_masks_minibox = torch.squeeze(roi_gt_masks_minibox, dim=1)
        # Threshold mask pixels at 0.5(have decimal cecause of RoiAlign) to have GT masks be 0 or 1
        # to use with binary cross entropy loss.
        roi_gt_masks_minibox = torch.round(roi_gt_masks_minibox)

        # Append negative ROIs and pad zeros for negative ROIs' bbox deltas and masks.
        rois = torch.cat([positive_rois, negative_rois], dim=0)
        n_nagetvie = negative_rois.shape[0]
        n_padding = torch.tensor(max(self.train_rois_per_image - rois.shape[0], 0))
        # Padding
        rois = torch.nn.functional.pad(rois, pad=[0, 0, 0, n_padding])
        roi_gt_boxes = torch.nn.functional.pad(roi_gt_boxes, pad=[0, 0, 0, n_padding+n_nagetvie])
        roi_gt_class_ids = torch.nn.functional.pad(roi_gt_class_ids, pad=[0, n_padding+n_nagetvie])
        deltas = torch.nn.functional.pad(deltas, pad=[0, 0, 0, n_padding+n_nagetvie])
        roi_gt_masks_minibox = torch.nn.functional.pad(roi_gt_masks_minibox, pad=[0, 0, 0, 0, 0, n_padding+n_nagetvie])

        return rois, roi_gt_class_ids, deltas, roi_gt_masks_minibox

    def normalize_to_mini_mask(self, rois, roi_gt_boxes):
        """
        Transform ROIs coordinates from normalized image space to gt boxes space which contains masks.
        i.e. Take gt box's bottom-left corner as (0,0) and upper-left corner as (1,1)

        rois:(N, [y1, x1, y2, x2]) in normalized coordinates (image space).
        roi_gt_boxes:(N, [gt_y1, gt_x1, gt_y2, gt_x2]) in normalized coordinates (image space).

        return:(N, [y1, x1, y2, x2]) in normalized coordinates (gt_boxes space).
        """
        y1, x1, y2, x2 = torch.split(rois, 1, dim=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = torch.split(roi_gt_boxes, 1, dim=1)

        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w

        boxes = torch.cat([y1, x1, y2, x2], dim=1)
        return boxes


# Pyramid ROI Align
class PyramidROIAlign:
    def __init__(self, pool_size, image_shape):
        self.pool_size = pool_size
        self.image_shape = image_shape

    def process(self, boxes, feature_maps):
        """
        boxes: (batch, n_boxes, [y1, x1, y2, x2]) in normalized coordinates
        feature_maps: [p2, p3, p4, p5], Each is (batch, channels, h, w). Note h and w is different among feature maps.

        return: (batch, n_boxes, channels, pool_size, pool_size). Box feature maps applying pyramid ROI align.
        """
        # Get boxes shapes
        y1, x1, y2, x2 = torch.split(boxes, 1, dim=2)
        h = y2 - y1
        w = x2 - x1

        # Equation 1 in the Feature Pyramid Networks paper.
        # Divides sqrt(image_area) because it's normalized coordinates here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = self.image_shape[0] * self.image_shape[1]
        k = 4 + torch.log2(torch.sqrt(h * w)/(224.0/math.sqrt(image_area)))
        # Should <=5 and >=2
        roi_level = torch.min(torch.tensor(5, dtype=torch.int32),
                              torch.max(torch.tensor(2, dtype=torch.int32), torch.round(k).to(torch.int32)))
        roi_level = torch.squeeze(roi_level, dim=2)

        # Loop through p2 to p5 and apply ROI polling to corresponding boxes.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            bool = torch.eq(roi_level, level)
            level_boxes = boxes[bool]
            ix = torch.nonzero(bool)

            # Keep track of which box is mapped to which level. (n_level_boxes, [batch_indice, box_indice])
            box_to_level.append(ix)

            # Box indices explaining which batch is belongs to for crop_and_resize. (n_level_boxes, [batch_indice])
            box_indices = ix[:, 0].to(torch.int32)

            # TODO: Need to stop gradient propogation to ROI proposals?
            # Crop and resize(ROI Align)
            level_boxes_transformed = transform_coordianates(level_boxes, self.image_shape)
            roi_align = RoIAlign(self.pool_size, self.pool_size)
            pooled.append(roi_align(feature_maps[i], level_boxes_transformed, box_indices))

        # (batch*n_boxes, channels, pool_size, pool_size)
        pooled = torch.cat(pooled, dim=0)
        # (batch*n_boxes, [batch_indice, box_indice])
        box_to_level = torch.cat(box_to_level, dim=0)

        # Rearrange pooled features to match the order of the original boxes.(different way to origin code)
        # Sort box_to_level by batch then box index
        # torch doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = torch.sort(sorting_tensor, dim=0)[1]
        pooled = torch.index_select(pooled, dim=0, index=ix)

        # Re-add the batch dimension. (batch, n_boxes, channels, pool_size, pool_size)
        shape = boxes.shape[:2] + pooled.shape[1:]
        pooled = torch.reshape(pooled, shape)
        return pooled
