import torch
from torch.utils.data import DataLoader
from dataset.coco import COCODataset
from model import Model
from model.Visualization import Observer
from Evaluation.Measurement import compute_box_ap
from model import Utils
import os
import numpy as np


device = torch.device("cuda:2, 3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
save_path = r'/home/yuruiqi/PycharmProjects/Mask_RCNN/save/try_coco.pkl'
test_data_dir = r'/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train2017_cat_dog'
img_shape = [512, 512]

# Get test data
test_data = COCODataset(test_data_dir)
test_loader = DataLoader(test_data, batch_size=9, shuffle=False)

# Load
mrcnn = Model.MRCNN(1, img_shape, n_classes=2, mode='inference', pretrain=False)
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

        # Test
        detection_boxes, detection_classes, detection_scores, mrcnn_masks = mrcnn(images)

        # Show
        observer = Observer(images, boxs, class_ids, rois, '/home/yuruiqi/visualization')
        observer.show_dataset()
        observer.get_detections(detection_boxes, detection_scores, detection_classes, mrcnn_masks)
        observer.show_detections(channel=0)
        observer.show_boxes(channel=0, boxes=mrcnn.vfm['rpn_rois'], save_dir=r'/home/yuruiqi/visualization/rpn')
        observer.show_boxes(channel=0, boxes=mrcnn.vfm['rpn_anchors'],
                            save_dir=r'/home/yuruiqi/visualization/rpn_anchors', scores=mrcnn.vfm['rpn_scores'])
        # observer.show_boxes(channel=0, boxes=mrcnn.vfm['refined_rois'], save_dir=r'/home/yuruiqi/visualization/refined_rpn')

        # visualize amchors and target boxes and masks
        mrcnn.set_train_gt(class_ids, boxs, rois)
        target_rpn_match, target_rpn_bbox = mrcnn.get_rpn_targets()
        # visualize origin anchor
        observer.show_boxes_filt(channel=0,boxes=mrcnn.anchors,
                                 match=target_rpn_match, match_score=1, save_dir=r'/home/yuruiqi/visualization/anchor')
        # visualize target
        observer.show_boxes_filt(channel=0, boxes=Utils.batch_slice([mrcnn.anchors, target_rpn_bbox], Utils.refine_boxes),
                                 match=target_rpn_match, match_score=1, save_dir=r'/home/yuruiqi/visualization/target_rpn_box')

        compute_box_ap(3, class_ids, boxs, detection_boxes, detection_classes, detection_scores)
        break
