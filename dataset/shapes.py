import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from model import Visualization, Utils
import os


class ShapeCreator:
    def __init__(self, image_shape, batch_size):
        self.image_shape = image_shape  # [h, w]
        self.shape_dict = {'circle': 1, 'rectangular': 2, 'diamond': 3}
        self.max_shape_per_img = 4

        self.canvas = np.zeros((batch_size, 1)+image_shape)  # (batch, channel, h, w)
        self.mask = np.zeros((batch_size, self.max_shape_per_img)+image_shape)  # (batch, max_shape_per_img, h, w)
        self.class_ids = np.zeros((batch_size, self.max_shape_per_img))  # (batch, max_shape_per_img)
        self.boxes = np.zeros((batch_size, self.max_shape_per_img, 4))  # (batch, max_shape_per_img, [y1, x1, y2, x2])

        self.radius_range = min(image_shape[0]//8, image_shape[1]//8)
        self.half_width_range = min(image_shape[0]//8, image_shape[1]//8)
        self.half_center_line_range = min(image_shape[0]//8, image_shape[1]//8)
        self.intensity_range = 255

    def generate_shape(self):
        self.draw_circle((50,50), 20, intensity=1, batch=0, no=0)
        self.draw_rectangular((190,130), 30, 20, intensity=1, batch=0, no=1)
        self.draw_diamond((70,150), 15, intensity=1, batch=0, no=2)
        self.draw_diamond((100,100), 15, intensity=1, batch=0, no=3)

        self.draw_circle((50,50), 20, intensity=1, batch=1, no=0)
        self.draw_circle((80,90), 20, intensity=1, batch=1, no=1)
        self.draw_rectangular((120,50), 20, 15, intensity=1, batch=1, no=2)
        self.draw_diamond((50,150), 20, intensity=1, batch=1, no=3)

        self.draw_circle((190,130), 20, intensity=1, batch=2, no=0)
        self.draw_rectangular((100,130), 30, 20, intensity=1, batch=2, no=1)
        self.draw_rectangular((190,30), 30, 20, intensity=1, batch=2, no=2)
        self.draw_diamond((50,30), 30, intensity=1, batch=2, no=3)

        self.draw_circle((100,150), 15, intensity=1, batch=3, no=0)
        self.draw_rectangular((175,105), 15, 30, intensity=1, batch=3, no=1)
        self.draw_diamond((60,85), 15, intensity=1, batch=3, no=2)
        self.draw_circle((30,30), 15, intensity=1, batch=3, no=3)

    def generate_random_shape(self, seed=19970516):
        np.random.seed(seed)
        # batch loop
        for batch in range(self.canvas.shape[0]):
            # shape loop
            for no in range(self.max_shape_per_img):
                center_y = np.random.randint(5, self.image_shape[0])
                center_x = np.random.randint(5, self.image_shape[1])
                center = (center_y, center_x)

                radius = np.random.randint(10, self.radius_range)
                half_height = np.random.randint(10, self.half_width_range)
                half_width = np.random.randint(10, self.half_width_range)
                half_center_line = np.random.randint(10, self.half_center_line_range)

                # intensity = np.random.randint(0, self.intensity_range)
                intensity = 1

                process_list = ['draw_circle', 'draw_rectangular', 'draw_diamond']
                process = random.choice(process_list)
                if process == 'draw_circle':
                    self.draw_circle(center, radius, intensity, batch, no)
                elif process == 'draw_rectangular':
                    self.draw_rectangular(center, half_height, half_width, intensity, batch, no)
                elif process == 'draw_diamond':
                    self.draw_diamond(center, half_center_line, intensity, batch, no)

    def draw_circle(self, center, radius, intensity, batch, no):
        """
        center: [y, x]
        radius: int
        intensity: int
        batch: int
        no: int. Number in range max_shape_per_img.
        """
        for y in range(self.image_shape[0]):
            for x in range(self.image_shape[1]):
                if (y - center[0]) ** 2 + (x - center[1]) ** 2 < radius ** 2:
                    self.canvas[batch, 0, y, x] += intensity
                    self.mask[batch, no, y, x] = 1

        box = [(center[0]-radius)/self.image_shape[0], (center[1]-radius)/self.image_shape[1],
               (center[0]+radius)/self.image_shape[0], (center[1]+radius)/self.image_shape[1]]
        self.class_ids[batch, no] = self.shape_dict['circle']
        self.boxes[batch, no] = box

    def draw_rectangular(self, center, half_height, half_width, intensity, batch, no):
        """
        center: [y, x]
        height: int
        width: int
        intensity: int
        batch: int
        no: int. Number in range max_shape_per_img.

        return: [y1, x1, y2, x2]
        """
        for y in range(self.image_shape[0]):
            for x in range(self.image_shape[1]):
                if abs(y - center[0]) < half_height and abs(x - center[1]) < half_width:
                    self.canvas[batch, 0, y, x] += intensity
                    self.mask[batch, no, y, x] = 1
        box = [(center[0]-half_height)/self.image_shape[0], (center[1]-half_width)/self.image_shape[1],
               (center[0]+half_height)/self.image_shape[0], (center[1]+half_width)/self.image_shape[1]]
        self.class_ids[batch, no] = self.shape_dict['rectangular']
        self.boxes[batch, no] = box

    def draw_diamond(self, center, half_center_line, intensity, batch, no):
        """
        Note that it's square actually 2333333.

        center: [y, x]
        half_center_line: int
        intensity: int
        batch: int
        no: int. Number in range max_shape_per_img.

        return: [y1, x1, y2.x2]
        """
        for y in range(self.image_shape[0]):
            for x in range(self.image_shape[1]):
                if abs(y - center[0]) + abs(x - center[1]) < half_center_line:
                    self.canvas[batch, 0, y, x] += intensity
                    self.mask[batch, no, y, x] = 1
        box = [(center[0]-half_center_line)/self.image_shape[0], (center[1]-half_center_line)/self.image_shape[1],
               (center[0]+half_center_line)/self.image_shape[0], (center[1]+half_center_line)/self.image_shape[1]]
        self.class_ids[batch, no] = self.shape_dict['diamond']
        self.boxes[batch, no] = box

    def get_data(self, to_gpu=True):
        images, gt_class_ids, gt_masks, gt_boxes = self.canvas, self.class_ids, self.mask, self.boxes

        # Transform to 3 channels.
        images = np.concatenate([images, images, images], axis=1)

        if to_gpu:
            images = torch.tensor(images, dtype=torch.float32)  # (batch, n_classes, 1, 256, 256)
            gt_class_ids = torch.tensor(gt_class_ids, dtype=torch.float32)  # (batch, n_classes)
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)  # (batch, n_classes, 4)
            gt_masks = torch.tensor(gt_masks, dtype=torch.float32)  # (batch, n_classes, 256, 256)

            images = images.cuda()
            gt_class_ids = gt_class_ids.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_masks = gt_masks.cuda()
        return images, gt_class_ids, gt_masks, gt_boxes

    def visualize(self, save_dir):
        boxes = Utils.denorm_boxes(self.boxes, self.canvas.shape[-2:])

        for batch in range(boxes.shape[0]):
            image = self.canvas[batch, 0]
            for i in range(boxes.shape[1]):
                box = boxes[batch, i]
                class_id = int(round(self.class_ids[batch, i]))
                mask = self.mask[batch, i]

                save_path = os.path.join(save_dir, '{}_{}'.format(batch, i))
                Visualization.visualize_1_box(image, box, name=str(class_id), mask=mask, save_path=save_path)


if __name__ == '__main__':
    shape_creator = ShapeCreator((256, 256), batch_size=4)
    shape_creator.generate_shape()
    # shape_creator.draw_circle((200, 50), 20, intensity=1, batch=0, no=0)
    # shape_creator.draw_rectangular((100, 100), 30, 20, intensity=1, batch=0, no=1)
    # shape_creator.draw_diamond((150, 150), 15, intensity=1, batch=0, no=2)
    shape_creator.visualize(r'/home/yuruiqi/visualization/rua/')
    images, gt_class_ids, gt_masks, gt_boxes = shape_creator.get_data()

    # Visualization.visualize_boxes(images, gt_boxes, save_dir=r'/home/yuruiqi/visualization/rua/')
