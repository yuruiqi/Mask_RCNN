import numpy as np
import tensorflow as tf
import torch
import  torchvision
from roi_align import RoIAlign      # RoIAlign module
from roi_align import CropAndResize # crop_and_resize module




if __name__ == '__main__':
    roi_level = tf.convert_to_tensor([[1, 2, 3, 4],
                                      [2, 3, 3, 5]])
    level = 3
    ix = tf.where(tf.equal(roi_level, level))
    box_indices = tf.cast(ix[:, 0], tf.int32)
    print(roi_level.shape, ix.shape)
    print(ix)
    print(box_indices)

