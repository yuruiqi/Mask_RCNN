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
def visualize_1_box(image, box, score=None,name='Unknow', mask=None, save_path=None, show=False):
    """
    Show box on the image. Note that the box is in original coordiantes.

    image: (h, w)
    box: [y1, x1, y2, x2]
    score: float
    name: str
    mask: (h, w)
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
    if isinstance(box, torch.Tensor):
        box = box.cpu().detach().numpy()
    if isinstance(score, torch.Tensor):
        score = score.cpu().detach().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    image = image.astype(np.float32)

    # [y1, x1, y2, x2] to [x1, y1, w, h]
    # print(box)
    y1, x1, y2, x2 = box.tolist()
    plt_box = plt.Rectangle((x1, y1), x2-x1, y2-y1, color='green', fill=False, linewidth=2)

    plt.imshow(image, cmap='gray')
    plt.gca().add_patch(plt_box)

    score = round(score, 2) if score else '/'
    box_tag = '{} {}'.format(name, score)
    plt.gca().text(x1, y1-8, box_tag, color='white', bbox={'facecolor':'green', 'alpha':1.0})

    if mask is not None:
        mask = np.where(mask>0.5, 1.0, 0.0)
        if mask.shape != image.shape:
            mask = Image.fromarray(mask)
            h = int(y2 - y1)
            w = int(x2 - x1)
            if h == 0:
                print('rua')
                h += 1
            if w == 0:
                print('rua')
                w += 1
            mask = torchvision.transforms.functional.resize(mask, (h, w))

            mask_on_image = np.zeros(image.shape)
            mask_on_image[int(y1):int(y2), int(x1):int(x2)] = mask

            plt.contour(mask_on_image)
        else:
            plt.contour(mask)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

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
            score = scores[batch, i_box] if (scores is not None) else None
            class_id = str(int(class_ids[batch, i_box])) if (class_ids is not None) else None

            for channel in range(images.shape[1]):
                save_path = os.path.join(save_dir, 'batch{} i{} channel{}.png'.format(batch, i_box, channel)) \
                    if save_dir else None
                visualize_1_box(images[batch, channel], boxes[batch, i_box], score, name=class_id, save_path=save_path)


def visualize_rpn_targets(images, anchors, rpn_bbox, rpn_match, save_dir=None, view_batch=None, n_watch=None):
    target_rpn_box = Utils.batch_slice([anchors, rpn_bbox], Utils.refine_boxes)

    image_shape = images.shape[2:]

    ix = torch.eq(rpn_match, 1)
    target_rpn_box = target_rpn_box[ix]
    images = torch.index_select(images, index=ix.nonzero()[:,0], dim=0).squeeze(dim=1)

    if isinstance(target_rpn_box, torch.Tensor):
        target_rpn_box = target_rpn_box.cpu().detach().numpy()
    target_rpn_box = Utils.denorm_boxes(target_rpn_box, image_shape)

    for i in range(images.shape[0]):
        visualize_1_box(images[i], target_rpn_box[i],  save_path=os.path.join(save_dir, '{}.png'.format(i)))


#################
# Visualize mask
#################
def visualize_mask(mrcnn_masks, save_dir=None, view_batch=None, n_watch=None, n_class_watch=None):
    if isinstance(mrcnn_masks, torch.Tensor):
        mrcnn_masks = mrcnn_masks.cpu().detach().numpy()
    mrcnn_masks = mrcnn_masks.astype(np.float32)

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

            if class_id is None or class_id < 1:
                continue

            mask = masks[batch, i_box, class_id] if (masks is not None) else None

            visualize_1_box(images[batch, 0], boxes[batch, i_box], score, str(class_id), mask, save_path=save_path)


def visualize_dataset(images, boxes, class_ids, masks, save_dir):
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if isinstance(class_ids, torch.Tensor):
        class_ids = class_ids.cpu().detach().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().detach().numpy()

    boxes = Utils.denorm_boxes(boxes, images.shape[-2:])

    for batch in range(boxes.shape[0]):
        for channel in range(images.shape[1]):
            image = images[batch, channel]
            for i in range(boxes.shape[1]):
                box = boxes[batch, i]
                class_id = int(round(class_ids[batch, i]))
                mask = masks[batch, i]

                save_path = os.path.join(save_dir, '{}_{}_{}'.format(batch, channel, i))
                visualize_1_box(image, box, name=str(class_id), mask=mask, save_path=save_path)


#############
# Observer
#############
class Observer:
    def __init__(self, images, boxes, class_ids, masks, save_dir):
        """
        images: (batch, channel, h, w)
        boxes: (batch, max_instances, 4)
        class_ids: (batch, max_instances)
        masks: (batch, max_instances, h, w)
        """
        # Origin data
        self.images = self.get_numpy_if_tensor(images)
        self.image_shape = self.images.shape[-2:]
        self.batchsize = self.images.shape[0]
        self.boxes = self.get_numpy_if_tensor(boxes)
        self.class_ids = self.get_numpy_if_tensor(class_ids)
        self.masks = self.get_numpy_if_tensor(masks)

        # Detection result
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_class_ids = None
        self.detection_masks = None

        # Intermediate variable
        self.rpn_boxes = None

        # Save path
        self.dataest_savepath = os.path.join(save_dir, 'dataset')
        self.detection_savepath = os.path.join(save_dir, 'detection')

    def get_numpy_if_tensor(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        return data
    
    def get_slice_data(self, data, slice):
        if data is not None:
            output = data[slice]
        else:
            output = None
        return output

    def get_detections(self, detection_boxes, detection_scores, detection_class_ids, detection_masks):
        self.detection_boxes = self.get_numpy_if_tensor(detection_boxes)
        self.detection_scores = self.get_numpy_if_tensor(detection_scores)
        self.detection_class_ids = self.get_numpy_if_tensor(detection_class_ids)
        self.detection_masks = self.get_numpy_if_tensor(detection_masks)

    def draw_1_box(self, box, name=None, score=None, mask=None):
        """
         Draw a box on one image. Note that the box is in original coordinates.

         box: [y1, x1, y2, x2]
         score: float
         mask: (h, w)
         name: str
        """
        # [y1, x1, y2, x2] to [x1, y1, w, h]
        # print(box)
        y1, x1, y2, x2 = box.tolist()
        plt_box = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='green', fill=False, linewidth=2)

        plt.gca().add_patch(plt_box)

        score = round(score, 2) if score else '/'

        # if to show tag
        if (name is not None) and (score is not None):
            box_tag = '{} {}'.format(name, score)
            plt.gca().text(x1, y1 - 8, box_tag, color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

        # if to show mask
        if mask is not None:
            mask = np.where(mask > 0.5, 1.0, 0.0)
            # if is mini mask
            if mask.shape != self.image_shape:
                mask = Image.fromarray(mask)
                h = int(y2 - y1)
                w = int(x2 - x1)
                if h == 0:
                    print('rua')
                    h += 1
                if w == 0:
                    print('rua')
                    w += 1
                mask = torchvision.transforms.functional.resize(mask, (h, w))

                mask_on_image = np.zeros(self.image_shape)
                mask_on_image[int(y1):int(y2), int(x1):int(x2)] = mask

                plt.contour(mask_on_image)
            else:
                plt.contour(mask)

    def draw_boxes(self, batch, channel, boxes, class_ids=None, scores=None, masks=None, save_dir=None):
        """
        Draw all boxes on one image.
        batch: int
        channel: int
        boxes: (n_boxes, n_classes)
        class_ids: (n_boxes)
        scores: (n_boxes)
        masks: (n_boxes, h, w)
        save_dir: str
        """
        image = self.images[batch, channel]
        plt.imshow(image, cmap='gray')

        for n_box in range(boxes.shape[0]):
            score_name = self.get_slice_data(scores, n_box)
            class_id = self.get_slice_data(class_ids, n_box)
            mask = self.get_slice_data(masks, n_box)
            self.draw_1_box(boxes[n_box], class_id, score_name, mask)

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(batch, channel)))
        plt.figure()

    def show_detections(self, channel):
        for batch in range(self.batchsize):
            batch_denorm_boxes = Utils.denorm_boxes(self.detection_boxes[batch], self.image_shape)

            # gather masks
            batch_masks_raw = self.detection_masks[batch]  # (max_instances, n_classes, h_mask, w_mask)
            batch_class_ids = self.detection_class_ids[batch]  # (max_instances)
            # (max_instances, h_mask, w_mask)
            detection_masks = batch_masks_raw[np.arange(0, batch_masks_raw.shape[0]), batch_class_ids.astype(np.int64)]
            
            self.draw_boxes(batch, channel, batch_denorm_boxes, batch_class_ids,
                            self.detection_scores[batch], detection_masks, self.detection_savepath)

    def show_dataset(self):
        for batch in range(self.batchsize):
            denorm_boxes = Utils.denorm_boxes(self.boxes[batch], self.image_shape)
            for channel in range(self.images.shape[1]):
                self.draw_boxes(batch, channel, denorm_boxes, self.class_ids[batch],
                                masks=self.masks[batch], save_dir=self.dataest_savepath)

    def show_boxes(self, channel, boxes, save_dir, scores=None):
        """
        channel: int.
        boxes: (batch, n_boxes, 4)
        scores: (batch, n_boxes)
        """
        boxes = self.get_numpy_if_tensor(boxes)
        scores = self.get_numpy_if_tensor(scores)
        for batch in range(self.batchsize):
            denorm_boxes = Utils.denorm_boxes(boxes[batch], self.image_shape)
            score = self.get_slice_data(scores, batch)
            self.draw_boxes(batch, channel, denorm_boxes, scores=score, save_dir=save_dir)

    def show_boxes_filt(self, channel, boxes, match, match_score, save_dir):
        """
        channel: int.
        boxes: (batch, n_boxes, 4)
        match: (batch, n_boxes)
        save_dir: str
        """
        boxes = self.get_numpy_if_tensor(boxes)
        match = self.get_numpy_if_tensor(match)
        for batch in range(self.batchsize):
            ix = np.equal(match[batch], match_score).nonzero()
            boxes_match = boxes[batch][ix]

            denorm_boxes = Utils.denorm_boxes(boxes_match, self.image_shape)
            self.draw_boxes(batch, channel, denorm_boxes, save_dir=save_dir)
