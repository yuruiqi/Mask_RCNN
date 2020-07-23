import torch
import torch.nn as nn
import numpy as np
import math
import time


# Anchor Generator
class AnchorGenerator:
    """
    To generate anchors according to ratios, scales and pyramid sizes.
    """
    def __init__(self, scales, ratios):
        """
        scales: tuple (scale1, scale2, ...). It means the area of the anchor will be scale^2.
        ratios: tuple (ratio1, ratio2, ...). It means h/w of the anchor will be ratio.
        """
        self.scales = scales
        self.ratios = ratios
        self.base_anchors = self._generate_base_anchors()

    def get_anchors(self, pyramid_shapes, feature_strides, anchor_stride=1):
        """
        The main function. To get anchors relative to different pyramid sizes.

        pyramid_shapes: list [p2_shape, p3_shape, p4_shape, p5_shape]. Every shape is interpreted in a tuple type.
        feature_strides: list [p2_stride, p3_stride, p4_shape, p5_stride]. The distance of between two pixels on original
                       image projected from the feature map. It's relative to image_size/feature_map_size.
        anchor_stride: int. The stride between the two pixels on feature map to get anchors.

        return:
        """
        all_anchors = []
        for i in range(len(pyramid_shapes)):
            all_anchors.append(self._generate_pyramid_anchors(pyramid_shapes[i], feature_strides[i], anchor_stride))
        all_anchors = np.concatenate(all_anchors, axis=0)

        return all_anchors

    # generate base anchors
    def _generate_base_anchors(self):
        """
        stride: int. It means the distance between the nearby pixels in a feature map projected to the origin image.
        scales: tuple (scale1, scale2, ...). It means the area of the anchor will be scale^2.
        ratios: tuple (ratio1, ratio2, ...). It means h/w of the anchor will be ratio.

        return: np.array [n_scales*n_ratios, 4], dtype=float. It means the base anchors of 1 pixel.
        """
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                anchors.append(self._generate_1_anchor(scale, ratio))
        anchors = np.asarray(anchors, dtype=np.float32)

        return anchors

    def _generate_1_anchor(self, scale, ratio):
        # TODO: To confirm that the return should be [-w/2, -h/2, w/2, h/2] or [-h/2, -w/2, h/2, w/2]
        """
        generate 1 anchor by 1 scale and ratio.
        Take the center of the receptive field on the origin image as origin.

        return: [-h/2, -w/2, h/2, w/2] dtype=float. the coordinates of the bottom-left and upper-right point
        of the box ith size of w and h.
        """
        scale = float(scale)
        ratio = float(ratio)

        # Easy to infer this formula.
        w = scale / math.sqrt(ratio)
        h = ratio * w

        return [-h/2, -w/2, h/2, w/2]

    # generate pyramid anchors
    def _generate_pyramid_anchors(self, pyramid_shape, feature_stride, anchor_stride):
        """
        Generate anchors according to the base anchors and the pyramid feature map shape.

        pyramid_shape: tuple (h, w). Shape of a pyramid feature map.

        return:
        """
        y_count, x_count = pyramid_shape

        anchors = []
        # Note that pyramid anchors should be arranged in order of anchors, transverse, longitude. So the loop order
        # should be base_anchors, x, y.
        for y in range(0, y_count, anchor_stride):
            for x in range(0, x_count, anchor_stride):
                # feature_stride*(x,y) means the center coordinates of the target anchors. So to every single pixel on
                # the origin image, we can just shift the base n anchors to the target center to generate new anchors.
                shift = feature_stride * np.array([y, x, y, x])
                anchors.append(self.base_anchors + shift)
        anchors = np.concatenate(anchors, axis=0)
        return anchors


# Apply deltas to anchors
def refine_boxes(anchors, deltas):
    """Applies the given deltas to the anchors.
    anchors: (N, [y1, x1, y2, x2]). Anchors to update
    deltas: (N, [dy, dx, log(dh), log(dw)]). Refinements to apply
    """
    # Convert to h, w, y, x
    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    center_y = anchors[:, 0] + 0.5 * height
    center_x = anchors[:, 1] + 0.5 * width

    # Apply deltas
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * torch.exp(deltas[:, 2])
    width = width * torch.exp(deltas[:, 3])

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


# Compute deltas from boxes to gt boxes
def compute_deltas(box, gt_box):
    """
    Compute deltas from boxes to corresponding gt boxes. It's a graph function.

    box: (N, [y1, x1, y2, x2])
    gt_box: (N, [y1, x1, y2, x2])

    return: (N, [dy, dx, log(dh), log(dw)])
    """
    # Convert to h, w, y, x
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    # Compute deltas
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


# Normalize boxes
def norm_boxes(boxes, image_shape):
    # TODO: To shift?
    #  Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.
    """
    Convert boxes from pixel coordinates to normalized coordinates.
    boxes: (N, [y1, x1, y2, x2]) in pixel coordinates
    image_shape: [height, width]

    return: (N, [y1, x1, y2, x2]) in normalized coordinates
    """
    h, w = image_shape
    scale = np.array([h-1, w-1, h-1, w-1])
    normalized_boxes = np.divide(boxes, scale)
    return normalized_boxes


# Normalize boxes
def denorm_boxes(boxes, image_shape):
    # TODO: To shift?
    #  Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.
    """
    Convert boxes from pixel coordinates to normalized coordinates.
    boxes: (N, [y1, x1, y2, x2]) in normalized coordinates
    image_shape: [height, width]

    return: (N, [y1, x1, y2, x2]) in pixel coordinates
    """
    h, w = image_shape
    scale = np.array([h-1, w-1, h-1, w-1])
    denormed_boxes = np.round(np.multiply(boxes, scale))
    return denormed_boxes


# Clip boxes
def clip_boxes(boxes, window):
    """
    Clip the boxes and remove the part beyond the window boundary.

    boxes: (N, [y1, x1, y2, x2])
    windows: ([wy1, wx1, wy2, wx2])

    return: clipped boxes
    """
    wy1, wx1, wy2, wx2 = torch.split(window, 1)
    y1, x1, y2, x2 = torch.split(boxes, 1, dim=1)

    # under upper boundary(min) and above bottom boundary(max)
    y1 = torch.max(torch.min(y1, wy2), wy1)
    x1 = torch.max(torch.min(x1, wx2), wx1)
    y2 = torch.max(torch.min(y2, wy2), wy1)
    x2 = torch.max(torch.min(x2, wx2), wx1)
    clipped_box = torch.cat([y1, x1, y2, x2], dim=1)
    return clipped_box


# Batch slice operation
def batch_slice(inputs, graph_fn):
    """
    Some functions cannot processed on different batch slices separately. We need to split the batch and operate on each
    slice, and then combine them.

    inputs: list. The data and relative parameters to operate. Please put the data in the first place, so the batch size
           can be gotten from  inputs[0].shape[0].
    graph_fn: Function. The graph function to operate on every slice.

    return: The output data after processed.
    """
    # Avoid only 1 input and cause error in the loop.
    if not isinstance(inputs, list):
        inputs = [inputs]

    batch_size = inputs[0].shape[0]
    outputs = []
    for i in range(batch_size):
        # get one slice from the total batch
        inputs_slice = [x[i] for x in inputs]
        # apply graph function
        output_slice = graph_fn(*inputs_slice)
        # Avoid error when torch.stack
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    # [(output1_b1, output2_b1, ...), (output1_b2, output2_b2, ...), ...] to
    # [(output1_b1, output1_b2, ...), (output2_b1, output1_b2, ...), ...]
    outputs = list(zip(*outputs))
    result = [torch.stack(x, dim=0) for x in outputs]

    if len(result) == 1:
        result = result[0]

    return result


# Remove zero paddings
def trim_zero_graph(boxes):
    """
    Remove the zero paddings of the boxes.

    boxes: (N, 4)
    return:
        boxes: (n, 4)
        non_zeros_ix: (n). Indices of the non-zeros.
    """
    # If the sum of all the abs(coordinates) is zero, it's zero padding.
    non_zeros = torch.sum(torch.abs(boxes), dim=1)
    non_zeros_ix = torch.nonzero(non_zeros, as_tuple=False).squeeze(1)
    boxes = torch.index_select(boxes, dim=0, index=non_zeros_ix)
    return boxes, non_zeros_ix


# Compute overlaps
def compute_overlaps(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1: (N1, [b1_y1, b1_x1, b1_y2, b1_x2])
    boxes2: (N2, [b2_y1, b2_x1, b2_y2, b2_x2])

    return: (N1, N2)
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = torch.zeros((boxes1.shape[0], boxes2.shape[0])).to(boxes1.device)
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou_multi(box2, boxes1, area2[i], area1)

    return overlaps


def compute_iou_multi(box1, boxes2, box1_area, boxes2_area):
    """
    Same to 'compute_iou'.
    Note: the areas are passed in rather than calculated here for efficiency.
    Calculate once in the caller to avoid duplicate work.

    box1: [y1, x1, y2, x2]
    boxes2: (n, [y1, x1, y2, x2])
    box1_area: float. the area of 'box'
    boxes2_area: (n)

    :return:
    """
    # Calculate intersection areas
    y1 = torch.max(box1[0], boxes2[:, 0])
    y2 = torch.min(box1[2], boxes2[:, 2])
    x1 = torch.max(box1[1], boxes2[:, 1])
    x2 = torch.min(box1[3], boxes2[:, 3])
    intersection = torch.max(x2 - x1, torch.tensor(0.0, device=x1.device, dtype=x1.dtype)) * \
                   torch.max(y2 - y1, torch.tensor(0.0, device=y1.device, dtype=y1.dtype))
    union = box1_area + boxes2_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.

    box1: [b1_y1, b1_x1, b1_y2, b1_x2]
    box2: [b2_y1, b2_x1, b2_y2, b2_x2]
    return: float
    """
    # Compute intersection
    b1_y1, b1_x1, b1_y2, b1_x2 = torch.split(box1, 1)
    b2_y1, b2_x1, b2_y2, b2_x2 = torch.split(box2, 1)
    y1 = torch.max(b1_y1, b2_y1)
    x1 = torch.max(b1_x1, b2_x1)
    y2 = torch.min(b1_y2, b2_y2)
    x2 = torch.min(b1_x2, b2_x2)
    intersection = torch.mul(torch.max(x2 - x1, torch.tensor([0], dtype=torch.float32, device=x1.device)),
                             torch.max(y2 - y1, torch.tensor([0], dtype=torch.float32, device=x1.device)))

    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    return iou


# Get rpn targets (Not used yet)
def compute_backbone_shapes(backbone_name, image_shape, feature_strides=[4, 8, 16, 32, 64]):
    # Currently supports ResNet50 only
    assert backbone_name in ["resnet50"]

    pyramid_shapes = [[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
                      for stride in feature_strides]
    return pyramid_shapes


def get_active_class_ids(gt_class_ids, n_classes):
    """

    gt_class_ids: (batch, max_instances). From 1 to n_class. Have zero-padding.
    n_classes: int.

    return: (batch, num_classes)
            Note: num_classes includes background.
    """
    array = torch.zeros([gt_class_ids.shape[0], n_classes+1], dtype=torch.int32, device=gt_class_ids.device)

    # trim zero-padding
    ix = gt_class_ids.gt(0).nonzero()
    ix_fill = torch.stack([ix[:, 0], (gt_class_ids[ix[:, 0], ix[:, 1]]).to(torch.int64)], dim=1)
    array[ix_fill[:, 0], ix_fill[:, 1]] = torch.tensor(1, dtype=torch.int32, device=gt_class_ids.device)

    return array


class GradSaver:
    def __init__(self):
        self.grads = {}

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def print_grad(self, name):
        try:
            grad = self.grads[name]
            print(name, grad.max().item(), grad.min().item())
        except:
            pass


if __name__ == '__main__':
    # ag = AnchorGenerator([32, ], [0.5, 1])
    # print((ag.get_anchors([(10, 10), (20, 20)], [16, 32])).shape)

    print(compute_backbone_shapes("resnet50", [256, 256]))
