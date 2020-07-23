import torch
import torch.nn as nn
from torchvision import ops
from model import Utils
import math


# Proposal Layer
class ProposalLayer:
    """
    Receives anchor scores and selects a subset to pass as proposals to the second stage. Filtering is done based on
    anchor scores and non-max suppression to remove overlaps. It also applies bounding box refinement deltas to anchors.

    The num_anchors is calculated by sigma(Pn_H * Pn_W * anchors_per_pixel).
    The anchors is arranged in order of anchors, transverse, longitude.
    """

    def __init__(self, post_nms_rois, nms_threshold=0.7, pre_nms_limit=6000):
        """
        post_nms_rois: ROIs kept after non-maximum suppression. Default to be 1000.
        nms_threshold:  Threshold of IOU to perform nms. Boxes IoU > nm_threshold will be discarded. Default to be 0.7.
        pre_nms_limt: ROIs kept after tf.nn.top_k and before non-maximum suppression. Default to be 6000.
        """
        self.proposal_count = post_nms_rois
        self.nms_threshold = nms_threshold
        self.pre_nms_limit = pre_nms_limit

        self.vfm = {}

    def process(self, anchors, scores, deltas):
        """
        anchors: (batch, num_anchors, [y1, x1, y2, x2]) anchors in normalized coordinates
        scores: (batch, num_anchors, [bg prob, fg prob])
        deltas: (batch, num_anchors, [dy, dx, log(dh), log(dw)])
        """
        # (batch, num_anchors, [fg_prob])
        scores = scores[:, :, 1]
        # TODO: Bounding box refinement standard deviation on deltas?

        # Filter out top N(pre_nms_limit) rois according to the scores and get their indices.
        scores, ix = torch.topk(scores, k=self.pre_nms_limit, dim=-1, sorted=True)
        deltas = Utils.batch_slice([deltas, ix], lambda x, y: torch.index_select(x, dim=0, index=y))
        anchors = Utils.batch_slice([anchors, ix], lambda x, y: torch.index_select(x, dim=0, index=y))

        # Apply deltas to anchors to get refined boxes. [batch, N, (y1, x1, y2, x2)]
        boxes = Utils.batch_slice([anchors, deltas], lambda x, y: Utils.refine_boxes(x, y))

        # Clip boxes
        window = torch.tensor([0, 0, 1, 1], dtype=boxes.dtype).to(device=boxes.device)
        boxes = Utils.batch_slice(boxes, lambda x: Utils.clip_boxes(x, window))

        # nms
        proposals = Utils.batch_slice([boxes, scores], self.nms)
        return proposals

    def nms(self, boxes, scores):
        """
        Operate non_maximal_suppresion on boxes.
        boxes: (N, [y1, x1, y2, x2]) in normalized coordinates.
        scores: (N, [fg_probs])

        return:  Remained boxes after nms.
        """
        indices = ops.nms(boxes, scores, self.nms_threshold)
        proposals = torch.index_select(boxes, dim=0, index=indices)
        # Because torch can't limit the proposal_count, it should be realized by myself.
        if proposals.shape[0] > self.proposal_count:
            scores, ix = torch.topk(scores, k=self.proposal_count, dim=-1, sorted=True)
            proposals = torch.index_select(proposals, dim=0, index=ix)
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
        gt_class_ids: (batch, max_gt_instances)
        gt_boxes: (batch, max_gt_instances, [y1, x1, y2, x2]) in normalized coordinates.
        gt_masks: (batch, max_gt_instances, height, width) of boolean type
        proposal_positive_ratio: float. Percent of positive ROIs in all rois used to train classifier/mask heads.
        train_proposals_per_image: int. Number of ROIs per image to feed to classifier/mask heads. Default to be 200.
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
            rois: (n, 4). With zero paddings.
            roi_gt_class_ids: (n). With zero paddings.
            deltas: (n, 4). With zero paddings.
            roi_gt_masks_minibox: (n, mask_h, mask_w)
        """
        # Remove zero padding
        proposals, _ = Utils.trim_zero_graph(proposals)
        gt_boxes, non_zeros_ix = Utils.trim_zero_graph(gt_boxes)
        gt_class_ids = torch.index_select(gt_class_ids, dim=0, index=non_zeros_ix)
        gt_masks = torch.index_select(gt_masks, dim=0, index=non_zeros_ix)

        # Compute overlaps. overlaps (n_proposals, n_gt_boxes)
        overlaps = Utils.compute_overlaps(proposals, gt_boxes)

        # Determine positive and negative ROIs.
        # To every proposal, get the max IoU with all the gt boxes.
        proposal_iou_max, _ = torch.max(overlaps, dim=1)
        # Positive rois are those with >= 0.5 IoU with a GT box.
        # Negative rois are those with < 0.5 IoU with every GT box.
        positive_roi_bool = torch.gt(proposal_iou_max,
                                     torch.tensor([0.5], dtype=proposal_iou_max.dtype, device=proposal_iou_max.device))
        positive_ix = torch.nonzero(positive_roi_bool)
        negative_ix = torch.nonzero(~positive_roi_bool)

        # Subsample rois to make positive/all = proposal_positive_ratio
        # 1. Positive rois (proposal_positive_ratio * train_proposals_per_image, 4)
        positive_count = int(self.roi_positive_ratio * self.train_rois_per_image)
        # TODO: Need shuffle on positive_ix and negative_ix before index selecting.
        positive_ix = positive_ix[0:positive_count].squeeze(1)
        # 2. Negative rois ((1-1/proposal_positive_ratio)*positive_count, 4)
        # Should calculated by this formula because positive rois may be not enough.
        negative_count = int((1 / self.roi_positive_ratio - 1) * positive_count)
        negative_ix = negative_ix[0:negative_count].squeeze(1)
        # 3. Gather selected rois
        positive_rois = torch.index_select(proposals, dim=0, index=positive_ix)
        negative_rois = torch.index_select(proposals, dim=0, index=negative_ix)

        # Assign positive rois to corresponding GT boxes
        # positive overlaps: (n_positive, n_gt_boxes)
        positive_overlaps = torch.index_select(overlaps, dim=0, index=positive_ix)
        # roi_gt_box_assignment: (n_positive), best corresponding GT box ids of every ROI
        if positive_overlaps.shape[0] > 0:
            roi_gt_box_assignment = torch.argmax(positive_overlaps, dim=1).to(positive_overlaps.device)
        else:
            roi_gt_box_assignment = torch.tensor([], dtype=torch.int64).to(positive_overlaps.device)
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
        box_ids = torch.unsqueeze(torch.arange(0, roi_gt_masks.shape[0]), dim=1).to(roi_gt_masks.dtype).to(roi_gt_masks.device)
        positive_rois_transformed = torch.cat([box_ids, positive_rois_transformed], dim=1)
        roi_gt_masks_minibox = ops.roi_align(roi_gt_masks, positive_rois_transformed, self.mask_shape)
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

        # TODO: require grad?
        deltas = deltas.detach()
        roi_gt_masks_minibox = roi_gt_masks_minibox.detach()
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


# Detection Layer
class DetectionLayer:
    """
    Return final detection boxes.
    """
    def __init__(self, detection_max_instances=100, detection_nms_threshold=0.3):
        self.window = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        self.detection_max_instance = detection_max_instances
        self.detection_nms_threshold = detection_nms_threshold

    def process(self, rois, mrcnn_class, mrcnn_bbox):
        """
        rois: (batch, n_rois, 4)
        mrcnn_class: (batch, n_rois, n_classes)
        mrcnn_bbox: (batch, n_rois, n_classes, 4)

        return: (batch, detection_max_instance, [y1, x1, y2, x2, class_id, score])
        """
        detections_batch = Utils.batch_slice([rois, mrcnn_class, mrcnn_bbox], self.refine_detections_graph)
        return detections_batch

    def refine_detections_graph(self, rois, probs, deltas):
        """
        rois: (N, [y1, x1, y2, x2]) in normalized coordinates.
        probs: (N, num_classes). All class probabilities of each roi.
              Note: num_classes includes background.
        deltas: (N, num_classes, [dy, dx, log(dh), log(dw)]). Deltas to all class of each roi.

        return: (detection_max_instance, [y1, x1, y2, x2, class_id, score])
        """
        # Best corresponding class to each roi.(from 0 to n_classes)
        class_ids = torch.argmax(probs, dim=1)  # (N)
        # Best corresponding class scores and deltas.
        class_scores = probs[torch.arange(class_ids.shape[0]), class_ids]  # (N)
        deltas_specific = deltas[torch.arange(class_ids.shape[0]), class_ids, :]  # (N,4)

        # Apply bounding box deltas TODO: deltas_specific * config.BBOX_STD_DEV?
        refined_rois = Utils.refine_boxes(rois, deltas_specific)
        refined_rois = Utils.clip_boxes(refined_rois, self.window.to(device=refined_rois.device))

        # Filter out background.
        keep = torch.nonzero(torch.gt(class_ids, 0))[:, 0]  # (n)
        # Omit filter out low confidence boxes. TODO: Confirm if it's appropriate.

        # Apply per-class NMS
        # 1. Prepare
        pre_nms_class_ids = class_ids[keep]  # (n)
        pre_nms_scores = class_scores[keep]  # (n)
        pre_nms_rois = refined_rois[keep]  # (n,4)
        unique_pre_nms_class_ids = torch.unique(pre_nms_class_ids)  # (n_unique). set of the class ids.

        def nms(class_id):
            """
            Apply Non-Maximum Suppression on ROIs of the given class.

            class_id: int.

            return: (detection_max_instance)
            """
            # Indices of ROIS of the given class
            ixs = torch.nonzero(torch.eq(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS. class_keep is the indice of the roi(after ixs index) after nms.
            class_keep = ops.nms(pre_nms_rois[ixs], pre_nms_scores[ixs], iou_threshold=self.detection_nms_threshold)
            # Because torch can't limit the proposal_count, it should be realized by myself.
            if class_keep.shape[0] > self.detection_max_instance:
                class_keep = class_keep[0:self.detection_max_instance]  # because ops.nms has sorted it.
            class_keep = keep[ixs[class_keep]]
            # Pad with -1 so it can stack over ids.
            padding_count = self.detection_max_instance - class_keep.shape[0]
            class_keep = nn.functional.pad(class_keep, [0, padding_count], value=-1)
            return class_keep

        # 2. Loop over class IDs. (n_unique, detection_max_instance)
        nms_keep = Utils.batch_slice(torch.unsqueeze(unique_pre_nms_class_ids, dim=1), nms)
        # 3. Merge results into one dim, and remove -1 padding.
        nms_keep = torch.reshape(nms_keep, [-1])  # (n_unique * detection_max_instance)
        nms_keep = nms_keep[torch.gt(nms_keep, -1)]  # (n_nms)

        # 4. Compute intersection between keep and nms_keep. TODO: why not just use nms_keep.
        keep = set(keep.cpu().numpy().tolist()).intersection(set(nms_keep.cpu().numpy().tolist()))
        keep = torch.tensor(list(keep)).to(nms_keep.device)

        # Keep top detections. TODO: redundant?
        class_scores_keep = class_scores[keep]
        num_keep = min(class_scores_keep.shape[0], self.detection_max_instance)
        top_ids = torch.topk(class_scores_keep, k=num_keep, sorted=True)[1]

        # Arrange output as (n_detections, [y1, x1, y2, x2, class_id, score])
        # TODO: Need add class_id by 1 because class_id is in [0, n_class-1]?
        detections = torch.cat([refined_rois[keep],
                                class_ids[keep].to(refined_rois.dtype).unsqueeze(dim=1),
                                torch.unsqueeze(class_scores[keep], dim=1)], dim=1)
        # Pad with zeros. Negative padding_count will reduce detections number to detection_max_instance.
        padding_count = self.detection_max_instance - detections.shape[0]
        detections = nn.functional.pad(detections, [0,0,0,padding_count])
        return detections


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
        # Get boxes shapes.py
        y1, x1, y2, x2 = torch.split(boxes, 1, dim=2)
        h = y2 - y1
        w = x2 - x1

        # Equation 1 in the Feature Pyramid Networks paper.
        # Divides sqrt(image_area) because it's normalized coordinates here.
        # e.g. a 224x224 ROI (in pixels) maps to P4.
        # TODO: 224 for an ROI is to large. Maybe I can change it.
        image_area = self.image_shape[0] * self.image_shape[1]
        k = 4 + torch.log2(torch.sqrt(h * w)/(224.0/math.sqrt(image_area)))
        # Should <=5 and >=2
        roi_level = torch.min(torch.tensor(5, dtype=torch.int16, device=k.device),
                              torch.max(torch.tensor(2, dtype=torch.int16, device=k.device), torch.round(k).to(torch.int16)))
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

            # Box indices explaining which batch it belongs to for crop_and_resize. (n_level_boxes, [batch_indice])
            box_indices = torch.unsqueeze(ix[:, 0].to(level_boxes.dtype), dim=1)

            # Need to stop gradient propogation to ROI proposals?
            level_boxes = level_boxes.detach()
            box_indices = box_indices.detach()

            # Crop and resize(ROI Align)
            feature_map_shape = (feature_maps[i].shape[-2], feature_maps[i].shape[-1])
            level_boxes_transformed = transform_coordianates(level_boxes, feature_map_shape)
            level_boxes_transformed = torch.cat([box_indices, level_boxes_transformed], dim=1)
            pooled.append(ops.roi_align(feature_maps[i], level_boxes_transformed, self.pool_size))

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
