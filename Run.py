import torch
import torch.nn as nn
from model.Model import MRCNN
from model import LossFunction
from dataset.shapes import ShapeCreator
from model import Utils, Visualization
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# device = torch.device("cuda:0")
device_ids = [0,1]


image_shape = (256, 256)
# Get data
shape_creator = ShapeCreator(image_shape, batch_size=4)
shape_creator.generate_shape()
images, gt_class_ids, gt_masks, gt_boxes = shape_creator.get_data()

# model
mrcnn = MRCNN(1, image_shape, mode='train', gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks)

# train
# model_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2.pkl'
rpn_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2_rpn.pkl'
head_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2_head.pkl'

mrcnn = mrcnn.cuda()

mrcnn.load_state_dict(torch.load(rpn_path))
mrcnn.train_part(images, save_path=rpn_path, part='RPN', lr=0.01, epoch=100)
mrcnn.load_state_dict(torch.load(rpn_path))
mrcnn.train_part(images, save_path=rpn_path, part='RPN', lr=0.001, epoch=100)
mrcnn.load_state_dict(torch.load(rpn_path))
mrcnn.train_part(images, save_path=head_path, part='Head', lr=0.01, epoch=100)
mrcnn.load_state_dict(torch.load(head_path))
mrcnn.train_part(images, save_path=head_path, part='Head', lr=0.001, epoch=100)
