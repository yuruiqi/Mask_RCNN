# Mask_RCNN
Mask R-CNN model programmed by Pytorch.

## Data Form
The input and output form of the model every batch.

###Input:
'max_instance_per_img' should be defined by the max instance number of one image throughout the dataset.

    images: (batch, channel, h, w)
            The input images.
    gt_boxes: (batch, max_instance_per_img, [y1, x1, y2, x2]). Float.
            The boxes of the target instances. The box is interpreted in form of left-bottom(y1, x1) and 
            right-upper(y2, x2) point of the box. 
            Use zero paddings if there is not enough instances to an image.
            Note that it is (y, x) form but not (x, y), and in normalized coordinates.
    gt_class_ids: (batch, max_instance_per_img). Int.
            The categories of the target instances.
            Use zero paddings if there is not enough instances to an image.
            Note that the id should begin from 1.
    gt_masks: (batch, max_instance_per_img, h, w)
            The mask of the target instances. Should contain only 0 and 1.
            Use zero paddings if there is not enough instances to an image.
    
###Output:
'detection_max_instance' can be set by yourself, and can be different to 'max_instance_per_img'. 

    detection_boxes: (batch, detection_max_instance, [y1, x1, y2, x2])
            The boxes of the detections.
    detection_classes: (batch, detection_max_instance, [class_id])
            The categories of the detections.
    detection_scores: (batch, detection_max_instance, [score])
            The confidence score of the detections.
    mrcnn_masks: (batch, num_rois, n_classes, mask_h, mask_w)
            The masks of the detections.
            Note that the size of masks is different to the original images. Should align to the box size and then apply
            to the image.

### E.g.
There are three images with 1 channel in a batch. Resolution ratio is 256 * 256.<br>
To detect 'dog', 'cat', 'rat' in the images, take 1, 2, 3 as their class ids.<br>

Image1: have a dog, a cat and two rat.<br>
Image2: have a dog, and a cat.<br>
Image3: have a dog, and a rat.

    images: (3, 1, 256, 256)
    
    gt_boxes: (3, 4, [y1, x1, y2, x2])
               Array: [[[0.1, 0.2, 0.4, 0.3],
                        [0.3, 0.4, 0.7, 0.8],
                        [0.7, 0.8, 0.8, 0.9],
                        [0.2, 0.1, 0.3, 0.4]],
                       
                       [[0.5, 0.3, 0.7, 0.4],
                        [0.3, 0.5, 0.4, 0.8],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]],
                       
                       [[0.3, 0.2, 0.4, 0.3],
                        [0.3, 0.5, 0.7, 0.8],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]]
                        
    gt_class_ids: (3, 4). 
                   Array: [[1, 2, 3, 3],
                           [1, 2, 0, 0],
                           [3, 2, 0, 0]]
                           
    gt_masks: (3, 4, 256, 256). With zero paddings.
