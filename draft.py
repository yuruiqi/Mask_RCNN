import torch
import numpy as np
import tensorflow as tf
import torchvision.ops as ops
import torchvision.models.detection.mask_rcnn
from model import Utils
# from dataset.shapes import ShapeCreator
from model.FunctionLayers import transform_coordianates
import matplotlib.pyplot as plt
import os

tf.nn.sparse_softmax_cross_entropy_with_logits()

if __name__ == '__main__':
    # 1.
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    #
    # shape_creator = ShapeCreator((256, 256), batch_size=1)
    # shape_creator.draw_rectangular((50, 150), 30, 20, intensity=1, batch=0, no=0)
    # images, gt_class_ids, gt_masks, gt_boxes = shape_creator.get_data()
    #
    # trans_box = transform_coordianates(gt_boxes[:, 0], (256, 256))
    # box = torch.cat([torch.tensor([[0]]).to(torch.float32).cuda(), trans_box], dim=1)
    #
    # rua = ops.roi_align(images, box, output_size=[60,40])
    # rua = rua.squeeze(dim=1).cpu()
    # plt.imshow(rua[0, 0])
    # plt.show()

    # 2.
    # numpy_data = np.zeros([40000, 1], dtype=float)
    # cpu_data = torch.tensor(numpy_data)
    # gpu_data = cpu_data.cuda()
    #
    # print('gpu: ', torch.argmax(gpu_data, dim=1))
    # print('cpu: ', torch.argmax(cpu_data, dim=1))

    # 3.
    grad_saver = Utils.GradSaver()
    size = 10

    x = 0.1 * torch.ones([size, size], dtype=torch.float32, requires_grad=True).cuda()
    y = torch.ones([size, size], dtype=torch.float32, requires_grad=False).cuda()
    z = torch.nn.functional.binary_cross_entropy(x, y)

    # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
    x.register_hook(grad_saver.save_grad('x_grad'))
    z.register_hook(grad_saver.save_grad('z_grad'))
    z.backward()

    grad_saver.print_grad('x_grad')
    grad_saver.print_grad('z_grad')
    print('z', z.item())
