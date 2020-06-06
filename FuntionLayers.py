import numpy as np
import tensorflow as tf
import torch
from torchvision import ops
import Utils


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


class DetectionTargetLayer:
    """
    Subsamples proposals by splitting positive and negative proposals.
    """

    def __init__(self, gt_class_ids, gt_boxes, gt_masks, proposal_positive_ratio, train_proposals_per_image):
        """

        """
        self.gt_class_ids = gt_class_ids
        self.gt_boxes = gt_boxes
        self.gt_masks = gt_masks
        self.proposal_positive_ratio = proposal_positive_ratio
        self.train_proposals_per_image = train_proposals_per_image

    def process(self, proposals):
        """

        """
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
        # Positive proposals are those with >= 0.5 IoU with a GT box.
        # Positive proposals are those with < 0.5 IoU with every GT box.
        positive_proposal_bool = torch.gt(proposal_iou_max, torch.tensor([0.5], dtype=torch.float32))
        positive_proposal_ix = torch.nonzero(positive_proposal_bool)
        negative_proposal_ix = torch.nonzero(~positive_proposal_bool)

        # Subsample proposals to make positive/all = proposal_positive_ratio
        # 1. Positive proposals (proposal_positive_ratio * train_proposals_per_image, 4)
        positive_count = int(self.proposal_positive_ratio * self.train_proposals_per_image)
        # TODO: Need shuffle on positive_proposal_ix and negative_proposal_ix before index selecting.
        positive_proposal_ix = positive_proposal_ix[0:positive_count].squeeze(1)
        # 2. Negative proposals ((1-positive_ratio * train_proposals_per_image), 4)
        negative_count = int((1 - self.proposal_positive_ratio) * self.train_proposals_per_image)
        negative_proposal_ix = negative_proposal_ix[0:negative_count].squeeze(1)
        # 3. Gather selected rois
        positive_proposals = torch.index_select(proposals, dim=0, index=positive_proposal_ix)
        negative_proposals = torch.index_select(proposals, dim=0, index=negative_proposal_ix)
