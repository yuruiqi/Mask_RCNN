import numpy as np
import math


class AnchorGenerator:
    """
    To generate anchors according to ratios, scales and pyramid sizes.
    """
    def __init__(self, scales, ratios):
        """
        scales: tuple (scale1, scale2, ...). It means the area of the anchor will be scale^2.
        ratios: tuple (ratio1, ratio2, ...). It means h/w of the anchor will be ratio.
        """
        self.scales = scales
        self.ratios = ratios
        self.base_anchors = self.generate_base_anchors()

    def get_anchors(self, pyramid_shapes, feature_stride, anchor_stride=1):
        """
        The main function. To get anchors relative to different pyramid sizes.

        pyramid_shapes: list [p2_shape, p3_shape, ..., p5_shape]. Every shape is interpreted in a tuple type.
        feature_stride: int. The distance of between two pixels on original image projected from the feature map. It's
                       relative to image_size/feature_map_size.
        anchor_stride: int. The stride between the two pixels on feature map to get anchors.

        return:
        """
        all_anchors = []
        for pyramid_shape in pyramid_shapes:
            pass
            all_anchors.append(self.generate_pyramid_anchors(pyramid_shape, feature_stride, anchor_stride))
        all_anchors = np.concatenate(all_anchors, axis=0)

        return all_anchors

    # generate base anchors
    def generate_base_anchors(self):
        """
        stride: int. It means the distance between the nearby pixels in a feature map projected to the origin image.
        scales: tuple (scale1, scale2, ...). It means the area of the anchor will be scale^2.
        ratios: tuple (ratio1, ratio2, ...). It means h/w of the anchor will be ratio.

        return: np.array [n_scales*n_ratios, 4], dtype=float. It means the base anchors of 1 pixel.
        """
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                anchors.append(self.generate_1_anchor(scale, ratio))
        anchors = np.asarray(anchors)

        return anchors

    def generate_1_anchor(self, scale, ratio):
        """
        generate 1 anchor by 1 scale and ratio.
        Take the center of the receptive field on the origin image as origin.

        return: [-w/2, -h/2, w/2, h/2] dtype=float. the coordinates of the bottom-left and upper-right point
        of the box ith size of w and h.
        """
        scale = float(scale)
        ratio = float(ratio)

        # Easy to infer this formula.
        w = scale / math.sqrt(ratio)
        h = ratio * w

        return [-w/2, -h/2, w/2, h/2]

    # generate pyramid anchors
    def generate_pyramid_anchors(self, pyramid_shape, feature_stride, anchor_stride):
        """
        Generate anchors according to the base anchors and the pyramid feature map shapes.

        pyramid_shapes: tuple (h, w). Shape of a pyramid feature map.

        return:
        """
        y_count, x_count = pyramid_shape

        anchors = []
        # Note that pyramid anchors should be arranged in order of anchors, transverse, longitude. So the loop order
        # should be base_anchors, x, y.
        for y in range(0, y_count, anchor_stride):
            for x in range(0, x_count, anchor_stride):
                # feature_stride*(x,y) means the center coordinates of the target anchors. So to every single pixel on
                # the origin image, we can just shift the base n anchors to the target center to generate new anchors.
                shift = feature_stride * np.array([x, y, x, y])
                anchors.append(self.base_anchors + shift)
        anchors = np.concatenate(anchors, axis=0)
        return anchors


if __name__ == '__main__':
    ag = AnchorGenerator([32, ], [0.5, 1])
    print((ag.get_anchors([(10, 10), (20, 20)], 16)).shape)
