import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import Utils
import os


#############
# Observer
#############
class Observer:
    def __init__(self, images, boxes, class_ids, masks, save_dir, show_channel=None):
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

        self.show_channel = show_channel

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
        # Judge zero array
        if np.where(box!=0)[0].size == 0:
            return None

        # [y1, x1, y2, x2] to [x1, y1, w, h]
        # print(box)
        y1, x1, y2, x2 = box.tolist()
        plt_box = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='green', fill=False, linewidth=2)

        plt.gca().add_patch(plt_box)

        # if to show tag
        if name is not None:
            if score is not None:
                box_tag = '{} {:.2f}'.format(name, score)
            else:
                box_tag = '{}'.format(name)
            plt.gca().text(x1, y1 - 8, box_tag, color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

        # if to show mask
        if mask is not None:
            mask = np.where(mask > 0.5, 1.0, 0.0)
            # if is mini mask
            if mask.shape != self.image_shape:
                mask = Image.fromarray(mask)
                h = int(y2 - y1)
                w = int(x2 - x1)
                mask = torchvision.transforms.functional.resize(mask, (h, w))

                mask_on_image = np.zeros(self.image_shape)
                mask_on_image[int(y1):int(y2), int(x1):int(x2)] = mask

                plt.contour(mask_on_image)
            else:
                plt.contour(mask)

    def draw_boxes(self, batch, boxes, class_ids=None, scores=None, masks=None, save_dir=None):
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
        if self.show_channel is None:
            image = self.images[batch].astype(np.int)
            image = np.transpose(image, [1, 2, 0])
            plt.imshow(image)
        else:
            image = self.images[batch, self.show_channel]
            plt.imshow(image, cmap='gray')

        for n_box in range(boxes.shape[0]):
            score_name = self.get_slice_data(scores, n_box)
            class_id = self.get_slice_data(class_ids, n_box)
            mask = self.get_slice_data(masks, n_box)
            self.draw_1_box(boxes[n_box], class_id, score_name, mask)

        if save_dir is not None:
            if self.show_channel is None:
                plt.savefig(os.path.join(save_dir, '{}.png'.format(batch)))
            else:
                plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(batch, self.show_channel)))
        plt.figure()

    def show_detections(self):
        for batch in range(self.batchsize):
            batch_denorm_boxes = Utils.denorm_boxes(self.detection_boxes[batch], self.image_shape)

            # gather masks
            batch_masks_raw = self.detection_masks[batch]  # (max_instances, n_classes, h_mask, w_mask)
            batch_class_ids = self.detection_class_ids[batch]  # (max_instances)
            # (max_instances, h_mask, w_mask)
            detection_masks = batch_masks_raw[np.arange(0, batch_masks_raw.shape[0]), batch_class_ids.astype(np.int64)]
            
            self.draw_boxes(batch, batch_denorm_boxes, batch_class_ids,
                            self.detection_scores[batch], detection_masks, self.detection_savepath)

    def show_dataset(self):
        for batch in range(self.batchsize):
            denorm_boxes = Utils.denorm_boxes(self.boxes[batch], self.image_shape)

            self.draw_boxes(batch, denorm_boxes, self.class_ids[batch],
                                masks=self.masks[batch], save_dir=self.dataest_savepath)

            # for channel in range(self.images.shape[1]):
            #     self.draw_boxes(batch, channel, denorm_boxes, self.class_ids[batch],
            #                     masks=self.masks[batch], save_dir=self.dataest_savepath)

    def show_boxes(self, boxes, save_dir, scores=None):
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
            self.draw_boxes(batch, denorm_boxes, scores=score, save_dir=save_dir)

    def show_boxes_match(self, boxes, match, match_score, save_dir):
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
            self.draw_boxes(batch, self.show_channel, denorm_boxes, save_dir=save_dir)

    def show_boxes_filt(self, boxes, score, threshold, save_dir):
        """
        channel: int.
        boxes: (batch, n_boxes, 4)
        match: (batch, n_boxes)
        save_dir: str
        """
        boxes = self.get_numpy_if_tensor(boxes)
        score = self.get_numpy_if_tensor(score)
        for batch in range(self.batchsize):
            ix = np.where(score[batch]>threshold, True, False)
            boxes_match = boxes[batch][ix]

            denorm_boxes = Utils.denorm_boxes(boxes_match, self.image_shape)
            self.draw_boxes(batch, denorm_boxes, save_dir=save_dir)
