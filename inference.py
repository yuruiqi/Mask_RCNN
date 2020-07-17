import torch
import torch.nn as nn
import torchvision.ops as ops
from model.Model import MRCNN
from dataset.shapes import ShapeCreator
import matplotlib.pyplot as plt
import model.Utils as Utils
import model.Visualization as Visualization
from Evaluation.Measurement import compute_box_ap
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

# Get data
shape_creator = ShapeCreator((256, 256), batch_size=4)
# shape_creator.generate_shape()
shape_creator.draw_circle((100, 100), 20, intensity=1, batch=0, no=0)
shape_creator.draw_rectangular((70, 130), 30, 20, intensity=1, batch=1, no=0)
shape_creator.draw_diamond((70, 100), 15, intensity=1, batch=2, no=0)
shape_creator.draw_circle((70, 100), 15, intensity=1, batch=3, no=0)
shape_creator.draw_diamond((130, 70), 15, intensity=1, batch=3, no=1)
images, gt_class_ids, gt_masks, gt_boxes = shape_creator.get_data()

mrcnn = MRCNN(1, [256, 256], mode='inference')
mrcnn.load_state_dict(torch.load(r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2_head.pkl'))
mrcnn.cuda()

with torch.no_grad():
    mrcnn.eval()
    detection_boxes, detection_classes, detection_scores, mrcnn_masks = mrcnn(images)
    # mrcnn_masks = torch.where(mrcnn_masks>0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())

    # Visualization.visualize_boxes(images, mrcnn.vfm['rpn_rois'],
    #                               save_dir=r'/home/yuruiqi/visualization/rpn_box/', n_watch=20, view_batch=None)
    # Visualization.visualize_boxes(images, detection_boxes, detection_scores, detection_classes,
    #                               save_dir=r'/home/yuruiqi/visualization/detection_box/', n_watch=50, view_batch=None)
    # Visualization.visualize_mask(mrcnn_masks, save_dir='/home/yuruiqi/visualization/mask/')
    #
    Visualization.visualize_detection(images, detection_boxes, detection_scores, detection_classes, mrcnn_masks,
                                  save_dir=r'/home/yuruiqi/visualization/detection/', n_watch=50, view_batch=None)

    # Visualization.visualize_mask(mrcnn.vfm['fpn_mask_conv1'], save_dir='/home/yuruiqi/visualization/fpn_mask_conv1/',
    #                              n_watch=50)

    compute_box_ap(3, gt_class_ids, gt_boxes, detection_boxes, detection_classes, detection_scores)
