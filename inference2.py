import torch
from torch.utils.data import DataLoader
from dataset.coco import COCODataset
from model import Model
from model.Visualization import Observer
from Evaluation.Measurement import compute_box_ap
from model import Utils
import os
import numpy as np
from model.LossFunction import LossComputer


device = torch.device("cuda:2, 3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# save_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try_coco2.pkl'
save_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try_coco.pkl'
# test_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train2017_cat_dog'
test_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train_val2017_cat_dog'
# test_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/val2017_cat_dog'
img_shape = [800, 800]

# Get test data
test_data = COCODataset(test_data_dir, img_shape)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

# Load
mrcnn = Model.MRCNN(img_shape, n_classes=2, mode='inference', pretrain=False,
                    scales=(32, 64, 128, 256, 512), p4_box_size=224.0)
mrcnn.to(device)
mrcnn.load_state_dict(torch.load(save_path))

mrcnn.eval()
test_loss_list = []
with torch.no_grad():
    for images, class_ids, rois, boxs in test_loader:
        # Data
        images = images.to(torch.float32).to(device)
        class_ids = class_ids.to(torch.int32).to(device)
        rois = rois.to(torch.float32).to(device)
        boxs = boxs.to(torch.float32).to(device)

        observer = Observer(images, boxs, class_ids, rois, '/home/yuruiqi/visualization')
        # observer.show_dataset()

        # losscomputer = LossComputer(loss_part='Heads', rpn_train_anchors_per_image=256)
        # losscomputer.get_rpn_targets(mrcnn.anchors, boxs)
        # losscomputer.get_active_class_ids(class_ids, n_classes=2)

        # Test
        detection_boxes, detection_classes, detection_scores, mrcnn_masks = mrcnn(images)

        # Show
        observer.get_detections(detection_boxes, detection_scores, detection_classes, mrcnn_masks)
        # observer.show_boxes_filt(boxes=mrcnn.vfm['rpn_anchors'], score=mrcnn.vfm['rpn_scores'], threshold=0.5,
        #                     save_dir=r'/home/yuruiqi/visualization/rpn_anchors', )
        # observer.show_boxes(boxes=mrcnn.vfm['rpn_rois'], save_dir=r'/home/yuruiqi/visualization/rpn')
        observer.show_detections()

        compute_box_ap(2, class_ids, boxs, detection_boxes, detection_classes, detection_scores)
        break
