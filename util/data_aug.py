import albumentations as albu
import numpy as np
import cv2
from albumentations import *


def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        HorizontalFlip(),
        Rotate(),
        RandomBrightnessContrast(),
    ], p=p)


def create_transformer(transformations, images):
    target = {}
    for i, image in enumerate(images[1:]):
        target['image' + str(i)] = 'image'
    return albu.Compose(transformations, p=0.5, additional_targets=target)(image=images[0],
                                                                           image0=images[1],
                                                                           image1=images[2]
                                                                           )


def aug_img_lab_reg(img, lab, reg, p=0.5):
    images = [img, lab, reg]
    transformed = create_transformer(strong_aug(p=p),  images)

    return np.transpose(transformed['image'],[2,0,1]), np.transpose(transformed['image0'],[2,0,1]),\
           np.transpose(transformed['image1'],[2,0,1])


#
# def aug_img_lab_reg_resize(img, lab, in_size, p=0.5):
#     images = [img, lab]
#     transformed = create_transformer(aug_resize(in_size, p=p),  images)
#     # im = transformed['image']     # img
#     # im0 = transformed['image0']   # lab
#     # im1 = transformed['image1']   # reg
#     # np.transpose(img_as_np, [1, 2, 0]
#     return np.transpose(transformed['image'],[2,0,1]), np.transpose(transformed['image0'],[2,0,1])
#
#
# def aug_resize(in_size, p=.5):
#     return Compose([
#         RandomRotate90(),
#         Flip(),
#         Transpose(),
#         HorizontalFlip(),
#         RandomRotate90(),
#         Rotate(),
#         OneOf([Resize(p=0.2,height=in_size[1], width=in_size[2]),
#                RandomSizedCrop(((in_size[1], in_size[2])), p=0.2, height=in_size[1], width=in_size[2],interpolation=2),
#                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2)
#                ], p=0.2),], p=p)
