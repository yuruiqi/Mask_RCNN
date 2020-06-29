import torch
import tensorflow as tf
import torchvision.ops as ops
import torchvision.models.detection.mask_rcnn
from model import Utils
from dataset import shapes
from model.FunctionLayers import transform_coordianates
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

    shape_creator = shapes.ShapeCreator((256, 256), batch_size=1)
    shape_creator.draw_rectangular((190, 130), 30, 20, intensity=1, batch=0, no=0)
    images, gt_class_ids, gt_masks, gt_boxes = shape_creator.get_data()

    trans_box = transform_coordianates(gt_boxes[:, 0], (256, 256))
    box = torch.cat([torch.tensor([[0]]).to(torch.float32).cuda(), trans_box], dim=1)

    rua = ops.roi_align(images, box, output_size=[60,40])
    rua = rua.squeeze(dim=1).cpu()
    plt.imshow(rua[0])
    plt.show()
