import torch
from torch.utils.data import DataLoader
from model import Model
from model.Visualization import Observer
from dataset.coco import COCODataset
import os
import numpy as np
from tensorboardX import SummaryWriter
import shutil

#########
# Config
#########
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
device = torch.device("cuda:2, 3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
train_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train2017_cat_dog'
val_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train2017_cat_dog'
batch_size = 8
n_epoch = 1000
lr = 0.001
patience = 100
save_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try_coco.pkl'
load_weight = True
train_bn = True

# train_part = ['RPN']
# loss_part = 'RPN'
# train_part = ['FPN_heads']
# loss_part = 'FPN_heads'
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
train_data = COCODataset(train_data_dir)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Get val data
# val_data = TrainDataset(val_data_dir)
val_data = COCODataset(train_data_dir)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Load
mrcnn = Model.MRCNN(3, [512, 512], n_classes=2, mode='train', pretrain=True)
# mrcnn.half()
mrcnn.to(device)
# mrcnn = torch.nn.DataParallel(mrcnn, device_ids=[2, 3])
if load_weight:
    mrcnn.load_state_dict(torch.load(save_path))

min_loss = None
patience_now = 0

########
# Train
########
mrcnn.set_trainable(train_part, train_bn=train_bn)


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
        # observer = Observer(images, boxs, class_ids, rois, '/home/yuruiqi/visualization')
        # observer.show_dataset()
        # Data
        images = images.to(torch.float32).to(device)
        class_ids = class_ids.to(torch.int32).to(device)
        rois = rois.to(torch.float32).to(device)
        boxs = boxs.to(torch.float32).to(device)

        # Compute train loss
        train_loss, train_loss_dict = mrcnn.train_part(images, class_ids, boxs, rois, part=loss_part)
        train_loss_list.append(train_loss.item())

        for key in train_loss_dict.keys():
            if key in train_loss_dict_epoch.keys():
                train_loss_dict_epoch[key].append(train_loss_dict[key])
            else:
                train_loss_dict_epoch[key] = [train_loss_dict[key]]
        # Optimize
        train_loss.backward()
        optimizer.step()

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

            # Compute val loss
            val_loss, val_loss_dict = mrcnn.train_part(val_images, val_class_ids, val_boxs, val_rois, part=loss_part)
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
    if (not min_loss) or (np.isnan(min_loss)) or val_loss_avg < min_loss:
        print('save')
        min_loss = val_loss_avg
        torch.save(mrcnn.state_dict(), save_path)
        patience_now = 0
    else:
        patience_now += 1
        if patience_now > patience:
            break
