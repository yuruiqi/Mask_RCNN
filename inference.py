import torch
import torch.nn as nn
import torchvision.ops as ops
from model.Model import MRCNN
from dataset.artifical_data import generate_data
import matplotlib.pyplot as plt
import model.Utils as Utils
import model.Visualization as Visualization


def visualize_rpn():
    rpn_feature_maps = mrcnn.vfm["rpn_feature_maps"]
    p2, p3, p4, p5, p6 = [x.cpu().detach().numpy() for x in rpn_feature_maps]
    for i in range(3):
        plt.imshow(p2[0,i,...], cmap='gray')
        plt.show()
        plt.imshow(p3[0,i,...], cmap='gray')
        plt.show()
        plt.imshow(p4[0,i,...], cmap='gray')
        plt.show()
        plt.imshow(p5[0,i,...], cmap='gray')
        plt.show()
        plt.imshow(p6[0,i,...], cmap='gray')
        plt.show()


def visualize_fpn_mask():
    fpn_mask_conv1 = mrcnn.vfm['fpn_mask_conv1'].cpu().detach().numpy()
    fpn_mask_conv2 = mrcnn.vfm['fpn_mask_conv2'].cpu().detach().numpy()
    fpn_mask_conv3 = mrcnn.vfm['fpn_mask_conv3'].cpu().detach().numpy()
    fpn_mask_conv4 = mrcnn.vfm['fpn_mask_conv4'].cpu().detach().numpy()
    fpn_mask_deconv = mrcnn.vfm['fpn_mask_deconv'].cpu().detach().numpy()
    fpn_mask_conv1x1 = mrcnn.vfm['fpn_mask_conv1x1'].cpu().detach().numpy()
    for i in range(3):
        plt.imshow(fpn_mask_conv1[0, 0, i, ...], cmap='gray')
        plt.show()
        plt.imshow(fpn_mask_conv2[0, 0, i, ...], cmap='gray')
        plt.show()
        plt.imshow(fpn_mask_conv3[0, 0, i, ...], cmap='gray')
        plt.show()
        plt.imshow(fpn_mask_conv4[0, 0, i, ...], cmap='gray')
        plt.show()
        plt.imshow(fpn_mask_deconv[0, 0, i, ...], cmap='gray')
        plt.show()
        plt.imshow(fpn_mask_conv1x1[0, 0, i, ...], cmap='gray')
        plt.show()


def visualize_mrcnn_mask(mrcnn_masks):
    # (batch, n_rois, n_classes, pool_size*2, pool_size*2)
    for i in range(mrcnn_masks.shape[1]):
        mask_0 = mrcnn_masks[0,i,0].cpu().detach().numpy()

        plt.imshow(mask_0, cmap='gray')
        plt.show()


# Get data
images, gt_class_ids, gt_boxes, gt_masks = generate_data()
images = torch.tensor(images, dtype=torch.float32)  # (batch, n_classes, 1, 256, 256)
gt_class_ids = torch.tensor(gt_class_ids, dtype=torch.float32)  # (batch, n_classes)
gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)  # (batch, n_classes, 4)
gt_masks = torch.tensor(gt_masks, dtype=torch.float32)  # (batch, n_classes, 256, 256)
# get active_class_ids
# active_class_ids = torch.where(gt_class_ids == 0,
#                                torch.tensor(0, dtype=torch.int32), torch.tensor(1, dtype=torch.int32))

images = images.cuda()
gt_class_ids = gt_class_ids.cuda()
gt_boxes = gt_boxes.cuda()
gt_masks = gt_masks.cuda()
# active_class_ids = active_class_ids.cuda()

mrcnn = MRCNN(1, [256, 256], mode='inference')
mrcnn.load_state_dict(torch.load(r'/home/yuruiqi/PycharmProjects/PI-RADS/save/model1_para.pkl'))
mrcnn.cuda()

with torch.no_grad():
    detection_boxes, detection_classes, detection_scores, mrcnn_masks = mrcnn(images)
    # mrcnn_masks = torch.where(mrcnn_masks>0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())

    # Visualization.visualize_rpn()
    # visualize_fpn_mask()
    # visualize_mrcnn_mask(mrcnn_masks)
    Visualization.visualize_rpn_rois(images, mrcnn.vfm['rpn_rois'], n_watch=300)
    # Visualization.visualize_detection_box(images, detection_boxes, detection_scores)
