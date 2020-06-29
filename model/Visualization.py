import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import Utils
import os


#################
# Visualize box
#################
def visualize_1_box(image, box, score=None,name='Unknow', mask=None, save_path=None):
    """
    Show box on the image. Note that the box is in original coordiantes.

    image: (h, w)
    box: [y1, x1, y2, x2]
    score: float
    name: str
    mask: (h, w)
    """
    # [y1, x1, y2, x2] to [x1, y1, w, h]
    # print(box)
    y1, x1, y2, x2 = box.tolist()
    plt_box = plt.Rectangle((x1, y1), x2-x1, y2-y1, color='green', fill=False, linewidth=2)

    plt.imshow(image)
    plt.gca().add_patch(plt_box)

    if score:
        score = float(score)
        box_tag = '{} {:.2f}'.format(name, score)
        plt.gca().text(x1, y1-5, box_tag, color='white', bbox={'facecolor':'green', 'alpha':1.0})

    if mask is not None:
        mask = np.where(mask>0.5, 1.0, 0.0)
        mask = Image.fromarray(mask)
        h = int(y2 - y1)
        w = int(x2 - x1)
        mask = torchvision.transforms.functional.resize(mask, (h, w))

        mask_on_image = np.zeros(image.shape)
        mask_on_image[int(y1):int(y2), int(x1):int(x2)] = mask

        plt.contour(mask_on_image)

    if save_path:
        plt.savefig(save_path)
    # plt.show()
    plt.figure()


def visualize_boxes(images, boxes, scores=None, class_ids=None, save_dir=None, view_batch=None, n_watch=None):
    """
    Visualize the boxes.

    images: (batch, channel, h, w)
    boxes: (batch, n_boxes, [y1, x1, y2, x2]) in normalized coordinates.
    scores: (batch, n_boxes, [fg_score])
    class_ids: (batch, n_boxes)
    save_dir: str. dir to save figures.
    view_batch: int. The batch to watch.
    n_watch: int. Num of rois to watch. No more than H * W * anchors_per_location.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()
    if isinstance(class_ids, torch.Tensor):
        class_ids = class_ids.cpu().detach().numpy()

    if view_batch:
        images = np.expand_dims(images[view_batch], axis=0)
    if (not n_watch) or n_watch > boxes.shape[1]:
        n_watch = boxes.shape[1]

    image_shape = images.shape[2:]
    boxes = Utils.denorm_boxes(boxes, image_shape)

    # batch loop
    for batch in range(images.shape[0]):
        # box loop
        for i_box in range(n_watch):
            save_path = os.path.join(save_dir, '{}_{}.png'.format(batch, i_box)) if save_dir else None
            score = scores[batch, i_box] if (scores is not None) else None
            class_id = str(int(class_ids[batch, i_box])) if (class_ids is not None) else None
            visualize_1_box(images[batch, 0], boxes[batch, i_box], score, name=class_id, save_path=save_path)


def visualize_rpn_targets(images, anchors, rpn_bbox, rpn_match, save_dir=None, view_batch=None, n_watch=None):
    target_rpn_box = Utils.batch_slice([anchors, rpn_bbox], Utils.refine_boxes)

    ix = torch.eq(rpn_match, 1)
    target_rpn_box = target_rpn_box[ix].reshape((target_rpn_box.shape[0], -1, 4))

    visualize_boxes(images, target_rpn_box, save_dir=save_dir, view_batch=view_batch, n_watch=n_watch)


#################
# Visualize mask
#################
def visualize_mask(mrcnn_masks, save_dir=None, view_batch=None, n_watch=None, n_class_watch=None):
    if isinstance(mrcnn_masks, torch.Tensor):
        mrcnn_masks = mrcnn_masks.cpu().detach().numpy()

    if view_batch:
        mrcnn_masks = np.expand_dims(mrcnn_masks[view_batch], axis=0)
    if (not n_watch) or n_watch > mrcnn_masks.shape[1]:
        n_watch = mrcnn_masks.shape[1]
    if (not n_class_watch) or n_class_watch > mrcnn_masks.shape[2]:
        n_class_watch = mrcnn_masks.shape[2]

    # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
    for batch in range(mrcnn_masks.shape[0]):
        for roi in range(n_watch):
            for category in range(n_class_watch):
                mask = mrcnn_masks[batch, roi, category]

                plt.imshow(mask, cmap='gray')
                if save_dir:
                    save_path = os.path.join(save_dir, 'batch{} roi{} class{}.png'.format(batch, roi, category+1))
                    plt.savefig(save_path)
                # plt.show()
                plt.figure()


def visualize_target_mask(target_masks, save_dir=None):
    if isinstance(target_masks, torch.Tensor):
        target_masks = target_masks.cpu().detach().numpy()

    # (batch, n_rois, pool_size*2, pool_size*2)
    for batch in range(target_masks.shape[0]):
        for roi in range(target_masks.shape[1]):
            mask = target_masks[batch, roi]

            plt.imshow(mask, cmap='gray')
            if save_dir:
                save_path = os.path.join(save_dir, 'batch{} roi{}.png'.format(batch, roi))
                plt.savefig(save_path)
            plt.show()


#########################
# Visualize box and mask
#########################
def visualize_detection(images, boxes, scores=None, class_ids=None, masks=None,
                        save_dir=None, view_batch=None, n_watch=None):
    """
    Visualize the box and apply the mask onto the box.

    images: (batch, channel, h, w)
    boxes: (batch, n_boxes, [y1, x1, y2, x2]) in normalized coordinates.
    scores: (batch, n_boxes, [fg_score])
    class_ids: (batch, n_boxes)
    masks: (batch, n_rois, n_classes, pool_size*2, pool_size*2)

    save_dir: str. dir to save figures.
    view_batch: int. The batch to watch.
    n_watch: int. Num of rois to watch. No more than H * W * anchors_per_location.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()
    if isinstance(class_ids, torch.Tensor):
        class_ids = class_ids.cpu().detach().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().detach().numpy()

    if view_batch:
        images = np.expand_dims(images[view_batch], axis=0)
    if (not n_watch) or n_watch > boxes.shape[1]:
        n_watch = boxes.shape[1]

    image_shape = images.shape[2:]
    boxes = Utils.denorm_boxes(boxes, image_shape)
    # batch loop
    for batch in range(images.shape[0]):
        # box loop
        for i_box in range(n_watch):
            save_path = os.path.join(save_dir, '{}_{}.png'.format(batch, i_box)) if save_dir else None
            score = scores[batch, i_box] if (scores is not None) else None
            class_id = int(class_ids[batch, i_box]) if (class_ids is not None) else None
            mask = masks[batch, i_box, class_id-1] if (masks is not None) else None

            visualize_1_box(images[batch, 0], boxes[batch, i_box], score, str(class_id), mask, save_path=save_path)
