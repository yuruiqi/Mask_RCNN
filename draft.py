import numpy as np
import tensorflow as tf
import torch
import  torchvision
from roi_align import RoIAlign      # RoIAlign module
from roi_align import CropAndResize # crop_and_resize module
import Utils


if __name__ == '__main__':
    active_class_ids = tf.convert_to_tensor([[1,1,1],
                                             [1,1,0]])
    pred_class_ids = tf.convert_to_tensor([[0,1,2,2,1,0],
                                           [0,2,1,1,0,2]])
    pred_active = tf.gather(active_class_ids[1], pred_class_ids)
    print(pred_class_ids)
    print(pred_active)
