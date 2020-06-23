import torch
import torch.nn as nn
from model.Model import MRCNN
from model import LossFunction
from dataset.artifical_data import generate_data
from model import Utils, Visualization
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# device = torch.device("cuda:0")
device_ids = [0,1]

# Get data
images, gt_class_ids, gt_boxes, gt_masks = generate_data()
images = torch.tensor(images, dtype=torch.float32)  # (batch, n_classes, 1, 256, 256)
gt_class_ids = torch.tensor(gt_class_ids, dtype=torch.float32)  # (batch, n_classes)
gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)  # (batch, n_classes, 4)
gt_masks = torch.tensor(gt_masks, dtype=torch.float32)  # (batch, n_classes, 256, 256)

images = images.cuda()
gt_class_ids = gt_class_ids.cuda()
gt_boxes = gt_boxes.cuda()
gt_masks = gt_masks.cuda()

# model
mrcnn = MRCNN(1, [256, 256], mode='train', gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks)

# 'rua'
# (batch, n_anchors, 4)
# anchors = mrcnn.generate_anchors([[64, 64], [32, 32], [16, 16], [8, 8], [4, 4]], [4, 8, 16, 32, 64], 2).cuda()
# # (batch, n_anchors), (batch, n_anchors, [dy, dx, log(dh), log(dw)])
# target_rpn_match, target_rpn_bbox = mrcnn.get_rpn_targets()
# target_box = Utils.batch_slice([anchors, target_rpn_bbox], lambda x, y: Utils.refine_boxes(x, y))
# Visualization.visualize_rpn_rois(images, target_rpn_bbox, target_rpn_match)

# train
para_path = r'/home/yuruiqi/PycharmProjects/PI-RADS/save/model1_para.pkl'

mrcnn = mrcnn.cuda()
optimizer = torch.optim.SGD(mrcnn.parameters(), lr=0.001)

mrcnn.load_state_dict(torch.load(para_path))
mrcnn.train_part(images, save_path=para_path, part='RPN', optimizer=optimizer, epoch=50)
# mrcnn.load_state_dict(torch.load(para_path))
# mrcnn.train_part(images, save_path=para_path, part='Head', optimizer=optimizer, epoch=20)
# mrcnn.load_state_dict(torch.load(para_path))
# mrcnn.train_part(images, save_path=para_path, part='All', optimizer=optimizer, epoch=20)
