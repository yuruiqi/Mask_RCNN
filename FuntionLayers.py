import numpy as np
import tensorflow as tf
import torch
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
        window = torch.tensor([0, 0, 1, 1], dtype=torch.double)
        boxes = Utils.batch_slice(boxes, lambda x: Utils.clip_boxes(x, window))
