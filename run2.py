import torch
from torch.utils.data import DataLoader
from model import Model
from model.LossFunction import LossComputer
from model.Visualization import Observer
from dataset.coco import COCODataset
import os
import numpy as np
from tensorboardX import SummaryWriter
import shutil

#########
# Config
#########
# device = torch.device("cuda:2, 3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
train_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train2017_cat_dog'
val_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train_val2017_cat_dog'
n_classes = 2
img_shape = (800, 800)
scales = (32, 64, 128, 256, 512)
p4_box_size = 224.0
n_epoch = 1000
patience = 100

rpn_train_anchors_per_image = 256
lr = 0.001
mrcnn_mask_lr_ratio = 100

batch_size = 4
save_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try_coco.pkl'
load_weight = True
train_bn = True

# 1. train RPN
# mode = 'RPN'
# train_part = ['RPN']
# loss_part = 'RPN'
# 2. train Heads
# mode = 'train'
# train_part = ['FPN_heads']
# loss_part = 'FPN_heads'
# 3. train all
mode = 'train'
train_part = ['Heads', 'Backbone']
loss_part = 'Heads'

# Tensorboard
tb_dir = r'/home/yuruiqi/visualization/tensorboard'
tb_path = os.path.join(tb_dir, loss_part)
if os.listdir(tb_path):
    shutil.rmtree(tb_path)
    os.mkdir(tb_path)
tb_writer = SummaryWriter(tb_path)

##########
# Prepare
##########
# Get train data
train_data = COCODataset(train_data_dir, img_shape)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Get val data
val_data = COCODataset(train_data_dir, img_shape)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load
mrcnn = Model.MRCNN(img_shape, n_classes=n_classes, mode=mode, pretrain=True,
                    scales=scales, p4_box_size=p4_box_size)
# mrcnn.half()
mrcnn.to(device)
# mrcnn = torch.nn.DataParallel(mrcnn, device_ids=[2, 3])
if load_weight:
    mrcnn.load_state_dict(torch.load(save_path))

min_loss = 100
patience_now = 0

########
# Train
########
Model.set_trainable(mrcnn, train_part, train_bn=train_bn)


def get_train_paras(paras, base_lr, train_bn=True):
    list = []
    for para_name, para in paras:
        # Skip bn
        if not train_bn:
            if 'bn' in para_name:
                continue
        # Skip requires_grad=False
        if para.requires_grad:
            # 100 lr for fpn_mask
            if 'fpn_mask' in para_name:
                para_lr = base_lr * 100
            # 0.1 lr for mrcnn_bbox
            # if 'dense_bbox' in para_name:
            #     para_lr = base_lr * 0.1
            else:
                para_lr = base_lr
            list.append({'params': para, 'lr': para_lr})
    return list


# rua = get_train_paras(mrcnn.named_parameters(), lr)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, mrcnn.parameters()), lr=lr)
optimizer = torch.optim.SGD(get_train_paras(mrcnn.named_parameters(), lr, train_bn=train_bn))
for epoch in range(n_epoch):
    print()
    print('Epoch: {}'.format(epoch))
    ########
    # Train
    ########
    if train_bn:
        mrcnn.train()
    train_loss_list = []
    train_loss_dict_epoch = {}
    for images, class_ids, rois, boxs in train_loader:
        optimizer.zero_grad()
        # Data
        images = images.to(torch.float32).to(device)
        class_ids = class_ids.to(torch.int32).to(device)
        rois = rois.to(torch.float32).to(device)
        boxs = boxs.to(torch.float32).to(device)

        if mrcnn.__class__.__name__ == 'DataParallel':
            mrcnn.module.detection_target_layer.get_gt(class_ids, boxs, rois)
        else:
            mrcnn.detection_target_layer.get_gt(class_ids, boxs, rois)

        # Compute train loss
        rpn_logits, rpn_scores, rpn_bboxes, \
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        target_class_ids, target_bbox, target_mask, anchors = mrcnn(images)

        # observer = Observer(images, boxs, class_ids, rois, '/home/yuruiqi/visualization')
        # observer.show_dataset()

        losscomputer = LossComputer(loss_part=loss_part, rpn_train_anchors_per_image=rpn_train_anchors_per_image)
        losscomputer.get_rpn_targets(anchors, boxs)
        losscomputer.get_active_class_ids(class_ids, n_classes)
        train_loss, train_loss_dict = losscomputer.get_loss(rpn_logits, rpn_bboxes,
                                                            mrcnn_class_logits, mrcnn_bbox, mrcnn_mask,
                                                            target_class_ids, target_bbox, target_mask)

        # observer.show_boxes_filt(channel=0, boxes=anchors, match=losscomputer.target_rpn_match, match_score=0,
        #                          save_dir=r'/home/yuruiqi/visualization/anchor')

        train_loss_list.append(train_loss.item())
        for key in train_loss_dict.keys():
            if key in train_loss_dict_epoch.keys():
                train_loss_dict_epoch[key].append(train_loss_dict[key])
            else:
                train_loss_dict_epoch[key] = [train_loss_dict[key]]
        # Optimize
        train_loss.backward()
        optimizer.step()

    # observer = Observer(images, boxs, class_ids, rois, '/home/yuruiqi/visualization')
    # observer.show_boxes(channel=0, boxes=mrcnn.vfm['rpn_rois'], save_dir=r'/home/yuruiqi/visualization/rpn')

    train_loss_avg = np.mean(train_loss_list)

    for key in train_loss_dict_epoch.keys():
        train_loss_dict_epoch[key] = np.mean(train_loss_dict_epoch[key])
    print('Train:', train_loss_avg, train_loss_dict_epoch)

    ########
    # Val
    ########
    mrcnn.eval()
    val_loss_list = []
    val_loss_dict_epoch = {}
    with torch.no_grad():
        for val_images, val_class_ids, val_rois, val_boxs in val_loader:
            # Data
            val_images = val_images.to(torch.float32).to(device)
            val_class_ids = val_class_ids.to(torch.int32).to(device)
            val_rois = val_rois.to(torch.float32).to(device)
            val_boxs = val_boxs.to(torch.float32).to(device)

            if mrcnn.__class__.__name__ == 'DataParallel':
                mrcnn.module.detection_target_layer.get_gt(val_class_ids, val_boxs, val_rois)
            else:
                mrcnn.detection_target_layer.get_gt(val_class_ids, val_boxs, val_rois)

            # Compute val loss
            val_rpn_logits, val_rpn_scores, val_rpn_bboxes, \
            val_mrcnn_class_logits, val_mrcnn_class, val_mrcnn_bbox, val_mrcnn_mask, \
            val_target_class_ids, val_target_bbox, val_target_mask, val_anchors = mrcnn(val_images)

            val_losscomputer = LossComputer(loss_part=loss_part, rpn_train_anchors_per_image=rpn_train_anchors_per_image)
            val_losscomputer.get_rpn_targets(val_anchors, val_boxs)
            val_losscomputer.get_active_class_ids(val_class_ids, n_classes)
            val_loss, val_loss_dict = val_losscomputer.get_loss(val_rpn_logits, val_rpn_bboxes,
                                                                val_mrcnn_class_logits, val_mrcnn_bbox, val_mrcnn_mask,
                                                                val_target_class_ids, val_target_bbox, val_target_mask)

            val_loss_list.append(val_loss.item())
            for key in val_loss_dict.keys():
                if key in val_loss_dict_epoch.keys():
                    val_loss_dict_epoch[key].append(val_loss_dict[key])
                else:
                    val_loss_dict_epoch[key] = [val_loss_dict[key]]

    val_loss_avg = np.mean(val_loss_list)

    for key in val_loss_dict_epoch.keys():
        val_loss_dict_epoch[key] = np.mean(val_loss_dict_epoch[key])
    print('Val: ', val_loss_avg, val_loss_dict_epoch)

    tb_writer.add_scalars('train_loss_dict', train_loss_dict_epoch, epoch)
    tb_writer.add_scalars('val_loss_dict', val_loss_dict_epoch, epoch)
    tb_writer.add_scalars('loss', {'train': train_loss_avg, 'val': val_loss_avg}, epoch)

    # Save
    if epoch == 0:
        min_loss = val_loss_avg
    elif (epoch > 0) and (not np.isnan(val_loss_avg)) and (val_loss_avg < min_loss):
        print('save')
        min_loss = val_loss_avg
        torch.save(mrcnn.state_dict(), save_path)
        patience_now = 0
    else:
        patience_now += 1
        if patience_now > patience:
            break
