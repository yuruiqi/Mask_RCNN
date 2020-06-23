import SimpleITK as sitk
from MeDIT.UsualUse import Imshow3DArray
import numpy as np
import os


def normalize(data):
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data


def show(image_path, roi_path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    image = normalize(np.transpose(image,[1,2,0]))
    print(image.shape)
    roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
    roi = normalize(np.transpose(roi,[1,2,0]))
    Imshow3DArray(image, roi)


if __name__ == '__main__':
    dir = r'W:\PrcoessedData\PI-RADS\2012-2016-CA_formal_BSL^bai song lai ^^6698-8'
    image_path = os.path.join(dir, r't2_Resize.nii')
    roi_path = os.path.join(dir, r'roi0_Resize.nii')
    show(image_path, roi_path)
