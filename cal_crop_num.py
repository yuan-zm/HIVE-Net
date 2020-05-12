import math


def cal_crop_num_img(img_size, in_size):
    if img_size[0] % in_size[0] == 0:
        crop_n1 = math.ceil(img_size[0] / in_size[0]) + 1
    else:
        crop_n1 = math.ceil(img_size[0] / in_size[0])

    if img_size[1] % in_size[1] == 0:
        crop_n2 = math.ceil(img_size[1] / in_size[1]) + 1
    else:
        crop_n2 = math.ceil(img_size[1] / in_size[1])

    if img_size[2] % in_size[2] == 0:
        crop_n3 = math.ceil(img_size[2] / in_size[2]) + 1
    else:
        crop_n3 = math.ceil(img_size[2] / in_size[2])
    return crop_n1, crop_n2, crop_n3
