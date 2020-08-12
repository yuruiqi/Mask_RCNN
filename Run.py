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
mrcnn = MRCNN(1, image_shape, mode='train')

# train
# model_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2.pkl'
rpn_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2_rpn.pkl'
head_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try2_head.pkl'

mrcnn = mrcnn.cuda()

############
# Train RPN
############
mrcnn.load_state_dict(torch.load(rpn_path))
# mrcnn.set_trainable(['RPN'])
mrcnn.set_trainable(['FPN_heads'])
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, mrcnn.parameters()), lr=0.001)

mrcnn.train()
min_loss = None
for epoch in range(1000):
    optimizer.zero_grad()
    loss, loss_dict = mrcnn.train_part(images, gt_class_ids, gt_boxes, gt_masks, part='FPN_heads')
    print('Epoch {}: loss:{}, {}'.format(epoch, loss.item(), loss_dict))

    if (not min_loss) or loss < min_loss:
        print('save')
        min_loss = loss
        torch.save(mrcnn.state_dict(), rpn_path)

    loss.backward()
    optimizer.step()
