from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import h5py
import os
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from model.Utils import norm_boxes

# cat:17:1, dog:18:2
category_dict = {17:1, 18:2}
# image ids whose #instance more than 3
max_instances = 3
skip_ids = [179725, 421903, 434765, 70754, 269432, 516738, 568117, 345434, 23917, 300415, 26503, 38282, 169426,153559]


###########
# generate h5
###########
def pad(array, max_instance):
    """
    array: (n_instance, shape0, shape1, ...)
    max_instance: int
    return: (max_instance, shape0, shape1, ...)
    """
    ori_shape = array.shape

    if ori_shape[0] == max_instance:
        return array
    zero_pad = np.zeros((1,)+ori_shape[1:])
    for i in range(max_instance - ori_shape[0]):
        array = np.concatenate([array, zero_pad], axis=0)
    return array

def transform_norm_coco_bbox(boxes, img_shape):
    """
    box: (n_boxes, [x1, y1, w, h])
    img_shape: (h ,w)
    return: (n_boxes, y1, x1, y2, x2)
    """
    x1, y1, w, h = np.split(boxes, 4, axis=1)  # (n_boxes, 1)
    x2 = x1 + w
    y2 = y1 + h
    boxes = np.stack([y1, x1, y2, x2], axis=1).squeeze(axis=2)  # (n_boxes, 4, 1) to (n_boxes, 4)

    h_img, w_img = img_shape
    scale = np.array([h_img - 1, w_img - 1, h_img - 1, w_img - 1])
    normalized_boxes = np.divide(boxes, scale)
    return normalized_boxes


def generate_coco_h5(coco_dir, dataType, cat_names, save_dir):
    annFile = '{}/annotations/instances_{}.json'.format(coco_dir,dataType)

    # load coco
    coco = COCO(annFile)

    # get cat(17) and dog(18) imgs
    catIds = coco.getCatIds(catNms=cat_names)
    imgIds = coco.getImgIds(catIds=catIds)

    for index in range(len(imgIds)):
        img = coco.loadImgs(imgIds[index])[0]
        img_id = img['id']
        # skip dropped ids
        if img_id in skip_ids:
            continue
        print(img_id)

        I = io.imread(img['coco_url'])
        I = np.transpose(I, [2,0,1])

        # load annotations
        annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        boxes = np.stack([x['bbox'] for x in anns], axis=0)
        boxes = transform_norm_coco_bbox(boxes, img_shape=I.shape[-2:])
        masks = np.stack([coco.annToMask(x) for x in anns], axis=0)
        class_ids = np.stack([category_dict[x['category_id']] for x in anns], axis=0)
        if class_ids.shape[0]>max_instances:
            print(class_ids.shape)

        # pad
        if len(boxes) < max_instances:
            boxes = pad(boxes, max_instances)
            masks = pad(masks, max_instances)
            class_ids = pad(class_ids, max_instances)

        f = h5py.File(os.path.join(save_dir, '{}.h5'.format(img_id)), mode='w')

        f['image'] = I
        f['class_ids'] = class_ids
        f['boxes'] = boxes
        f['rois'] = masks
        f.close()


###########
# Dataset
###########
def transform_graph(image, size):
    """
    image: (channel, h, w). Note that channel should be 1.
    size: [H, W]

    return: (channel, H, W)
    """
    transform = torchvision.transforms.Scale(size)
    # (channel, h, w)
    n_channel = image.shape[0]
    image = np.stack([transform(Image.fromarray(image[c])) for c in range(n_channel)], axis=0)
    return image


def transform(images, size):
    """
    images: (n_images, channel, h, w). Note that channel should be 1.
    size: [H, W]

    return: (n_images, channel, H, W)
    """
    image_list = []
    for i in range(images.shape[0]):
        image = transform_graph(images[i], size)
        image_list.append(image)
    return np.stack(image_list, axis=0)


class COCODataset(Dataset):
    def __init__(self, set_dir, img_shape):
        # self.data = np.array([os.path.join(set_dir, x) for x in os.listdir(set_dir)])
        self.data_paths = [os.path.join(set_dir, x) for x in sorted(os.listdir(set_dir))]
        self.img_shape = img_shape

    def __getitem__(self, index):
        data_h5 = h5py.File(str(self.data_paths[index]), 'r')

        img = data_h5['image'].value
        class_ids = data_h5['class_ids'].value
        boxes = data_h5['boxes'].value
        masks = data_h5['rois'].value

        img = transform_graph(img, self.img_shape)
        masks = transform(masks[:, np.newaxis, ...], self.img_shape).squeeze(axis=1)

        data_h5.close()

        # print(self.data_paths[index])
        return img.astype(np.float32), class_ids.astype(np.int32), masks.astype(np.float32), boxes.astype(np.float32)

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    coco_dir = '/home/yuruiqi/PycharmProjects/COCOData'
    # dataType = 'train2017'
    # save_dir = '/home/yuruiqi/PycharmProjects/COCOData_mrcnn/train2017_cat_dog'
    # dataType = 'val2017'
    # save_dir = '/home/yuruiqi/PycharmProjects/COCOData_mrcnn/val2017_cat_dog'
    dataType = 'test2017'
    save_dir = '/home/yuruiqi/PycharmProjects/COCOData_mrcnn/test2017_cat_dog'

    generate_coco_h5(coco_dir, dataType, ['cat', 'dog'], save_dir)
