import torch
import numpy as np
from model import Utils
from collections import Counter
import matplotlib.pyplot as plt


def get_box_ap_dict_graph(n_class_ids, gt_class_ids, gt_boxes, detection_boxes, detection_classes, detection_scores):
    """
    Get ap tp/fp list of the detection boxes in an image.

    n_class_ids: int. The number of classes.
    gt_class_ids: (max_instance_per_img)
    gt_boxes: (max_instance_per_img, [y1, x1, y2, x2])
    detection_boxes: (detection_max_instance, [y1, x1, y2, x2])
    detection_classes: (detection_max_instance, [class_id])
    detection_scores: (detection_max_instance, [score])

    return:
        class_ap_dict: {class_id: [confidence, judge]}
    """
    # Create ap dict. {class_id: [confidence, judge]}
    class_ap_dict = {}
    for i in range(n_class_ids):
        class_ap_dict[i + 1] = []

    for class_id_from_zero in range(n_class_ids):
        class_id = class_id_from_zero + 1
        gt_index = gt_class_ids.eq(class_id)
        gt_box = gt_boxes[gt_index]  # (n_gt_box)

        detection_index = detection_classes.eq(class_id)
        confidence = detection_scores[detection_index]  # (n_detection_box)
        detection_box = detection_boxes[detection_index]  # (n_detection_box, 4)

        if gt_box.shape[0] == 0:
            tp_index = set()
        else:
            overlaps = Utils.compute_overlaps(detection_box, gt_box)  # (n_detection_box, n_gt_box)
            # 1. For every gt box, get the max IoU as tp.
            if overlaps.shape[0] > 1:
                tp_index1 = overlaps.argmax(dim=0)  # (n_gt_box)
            else:
                tp_index1 = torch.tensor([0], dtype=torch.int32)
            # 2. Get the index of the box which has IoU>0.5.
            tp_index2 = overlaps.gt(0.5).nonzero()[:,0]
            # 3. Take intersection set.
            tp_index1 = tp_index1.cpu().numpy().tolist()
            tp_index2 = tp_index2.cpu().numpy().tolist()
            tp_index = set(tp_index1).intersection(set(tp_index2))

        # Append [confidence, judge] for specific class_id.
        for n in range(confidence.shape[0]):
            if n in tp_index:
                judge = 'tp'
            else:
                judge = 'fp'
            class_ap_dict[class_id].append([confidence[n].cpu().item(), judge])
    return class_ap_dict


def compute_ap(precision, recall):
    """

    precision: list.
    recall: list. In ascending order.

    return:
    """
    # Pad with start and end values.
    precision = [0] + precision + [0]
    recall = [0] + recall + [1]

    # TODO: Comprehend.
    for i in range(len(precision)-1, 0, -1):
        precision[i-1] = max(precision[i-1], precision[i])

    i = np.where(recall[:-1] != recall[1:])[0].item()
    ap = np.sum((recall[i+1] - recall[i]) * precision[i+1])
    return ap


def compute_class_ap(class_ap_list, n_all_gt_class):
    """
    class_ap_list: [[confidence, judge], ...]
    n_all_gt_class:  int.

    return:
    """
    def take_first(elem):
        return elem[0]
    # Sort by confidence in descending order
    class_ap_list.sort(key=take_first, reverse=True)

    # Accumulate on tp and fp.
    tp_acc = 0
    fp_acc = 0
    precision = []
    recall = []
    for detection in class_ap_list:
        if detection[1] == 'tp':
            tp_acc += 1
        elif detection[1] == 'fp':
            fp_acc += 1
        else:
            raise ValueError('Rua')
        precision.append(tp_acc/(tp_acc + fp_acc))
        recall.append(tp_acc/n_all_gt_class)
    # print(recall, precision)
    # Compute AP
    plt.plot(recall, precision)
    plt.show()
    ap = compute_ap(precision, recall)
    return ap


def compute_box_ap(n_class_ids, gt_class_ids, gt_boxes, detection_boxes, detection_classes, detection_scores):
    """
    n_class_ids: int. The number of classes.
    gt_class_ids: (batch, max_instance_per_img)
    gt_boxes: (batch, max_instance_per_img, [y1, x1, y2, x2])
    detection_boxes: (batch, detection_max_instance, [y1, x1, y2, x2])
    detection_classes: (batch, detection_max_instance, [class_id])
    detection_scores: (batch, detection_max_instance, [score])

    return:
    """
    gt_class_ids = gt_class_ids.int()

    # Create ap dict. {class_id: [confidence, judge]}
    class_ap_dict = {}
    for i in range(n_class_ids):
        class_ap_dict[i+1] = []

    for batch in range(gt_class_ids.shape[0]):
        class_ap_dict_graph = get_box_ap_dict_graph(n_class_ids, gt_class_ids[batch], gt_boxes[batch],
                                                    detection_boxes[batch], detection_classes[batch], detection_scores[batch])
        # Merge dict
        for key in class_ap_dict.keys():
            class_ap_dict[key].extend(class_ap_dict_graph[key])

    # Get n of gt of specific class_id.
    n_all_gt_dict = Counter(gt_class_ids.cpu().numpy().reshape([-1]))
    if n_all_gt_dict[0]:
        n_all_gt_dict.pop(0)

    for class_id in n_all_gt_dict.keys():
        class_ap = compute_class_ap(class_ap_dict[class_id], n_all_gt_dict[class_id])
        print('class {}: {}'.format(class_id, class_ap))
