import torch
import torch.nn as nn
import h5py

from model.NetworkLayers import ResNet
from torchvision.models.resnet import resnet50, ResNet, Bottleneck


# resnet50_path = '/home/yuruiqi/PycharmProjects/Mask_RCNN/save/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
# resnet50_path = '/home/yuruiqi/PycharmProjects/Mask_RCNN/save/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


class ResNetPyTorch(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(Bottleneck, [3,4,6,3])

        resnet50_path = '/home/yuruiqi/PycharmProjects/Mask_RCNN/save/resnet50-19c8e357.pth'
        if pretrained:
            self.load_state_dict(torch.load(resnet50_path))

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return [c1, c2, c3, c4, c5]


rua = ResNetPyTorch(pretrained=True)
