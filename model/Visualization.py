import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Utils
import os


def visualize_1_box(image, box, score=None, name='Unknow', save_name=None):
    """
    Show box on the image.

    image: (h, w)
    box: [y1, x1, y2, x2] in normalized coordinates.
    score: float
    name: str
    """
    # [y1, x1, y2, x2] to [x1, y1, w, h]
    y1, x1, y2, x2 = box.tolist()
    plt_box = plt.Rectangle((x1, y1), x2-x1, y2-y1, color='green', fill=False, linewidth=2)

    plt.imshow(image)
    if save_name:
        save_dir = r'/home/yuruiqi/rua/'
        plt.savefig(os.path.join(save_dir, save_name))
    plt.show()
    plt.gca().add_patch(plt_box)

    if score:
        box_tag = '{} {}'.format(name, score)
        plt.gca().text(x1, y1, box_tag, color='white', bbox={'facecolor':'green', 'alpha':1.0})


def visualize_mrcnn_mask(mrcnn_masks):
    # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
    for i in range(mrcnn_masks.shape[1]):
        mask_0 = mrcnn_masks[0,i,0].cpu().detach().numpy()

        plt.imshow(mask_0, cmap='gray')
        plt.show()


def visualize_rpn_rois(images, rpn_rois, rpn_scores=None, view_batch=0, n_watch=30):
    """
    Visualize the output of RPN. Note that rpn_rois have been applied deltas by proposal layer.

    images: (batch, channel, h, w)
    rpn_rois: (batch, h * w * anchors_per_location, [y1, x1, y2, x2]) in normalized coordinates.
    rpn_scores: (batch, H * W * anchors_per_location, [bg_score, fg_score])
                or (batch, H * W * anchors_per_location, fg_score).
    view_batch: int. The batch to watch.
    n_watch: int. Num of rois to watch. No more than H * W * anchors_per_location.
    """
    images = images.cpu().detach().numpy()
    rpn_rois = rpn_rois.cpu().detach().numpy()

    image = images[view_batch, 0]
    rpn_rois = Utils.denorm_boxes(rpn_rois, image.shape)

    for i in range(n_watch):
            # [bg_score, fg_score] on last dim
        if rpn_scores:
            rpn_scores = rpn_scores.cpu().detach().numpy()
            if len(rpn_scores.shape) == 3:
                visualize_1_box(image, rpn_rois[view_batch, i],rpn_scores[view_batch, i, 1])
            # only fg_score on last dim. (To visualize target_rpn)
            else:
                visualize_1_box(image, rpn_rois[view_batch, i], rpn_scores[view_batch, i])
        else:
            visualize_1_box(image, rpn_rois[view_batch, i], save_name=str(i)+'.png')
            # visualize_1_box(image, rpn_rois[view_batch, i])

def visualize_detection_box(images, detection_boxes, detection_scores, view_batch=0):
    """
    Visualize the final detection box.

    images: (batch, channel, h, w)
    detection_boxes: (batch, detection_max_instance, [y1, x1, y2, x2]) in normalized coordinates.
    detection_scores: (batch, detection_max_instance, [score])
    view_batch: int. The batch to watch.
    """
    images = images.cpu().detach().numpy()
    detection_boxes = detection_boxes.cpu().detach().numpy()
    detection_scores = detection_scores.cpu().detach().numpy()

    image = images[view_batch, 0]
    detection_boxes = Utils.denorm_boxes(detection_boxes, image.shape)

    for i in range(detection_boxes.shape[1]):
        print(detection_boxes[view_batch, i], detection_scores[view_batch, i])
        visualize_1_box(image, detection_boxes[view_batch, i], detection_scores[view_batch, i])
