import nibabel, glob
import numpy as np
from util.tools_self import save_array_as_nii_volume, load_nifty_volume_as_array
from metrics import *


def save_com_img2nii(pred, lab, save_path):

    lab[lab == 255] = 5
    pred[pred == 255] = 1
    re = lab - pred
    re[re == 255] = 3
    re[re == 4] = 2
    re[re == 5] = 1

    print(np.unique(re[:]))
    save_array_as_nii_volume(re, save_path)


def approximate_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


lab_path = './oriCvLab/testLab.nii.gz'
target = load_nifty_volume_as_array(lab_path)
print(np.unique(target))
target = approximate_image(target)
print(np.unique(target))

test_epoch = 405
is_over_lap = 0

if is_over_lap == 0:
    flip_1_path = './history/RMS/result_images_test/epoch_' + str(test_epoch) + '/' + str(test_epoch) +'_pro.nii.gz'
    flip_2_path = './history/RMS/result_images_test_1/epoch_' + str(test_epoch) + '/' + str(test_epoch) +'_pro.nii.gz'
    flip_3_path = './history/RMS/result_images_test_2/epoch_' + str(test_epoch) + '/' + str(test_epoch) +'_pro.nii.gz'
elif is_over_lap != 0:
    flip_1_path = 'sliceTest/result_images_test/epoch_' + str(test_epoch) + '/' + str(test_epoch) + '_pro.nii.gz'
    flip_2_path = './sliceTest/result_images_test_1/epoch_' + str(test_epoch) + '/' + str(test_epoch) + '_pro.nii.gz'
    flip_3_path = './sliceTest/result_images_test_2/epoch_' + str(test_epoch) + '/' + str(test_epoch) + '_pro.nii.gz'

pro1 = load_nifty_volume_as_array(flip_1_path)
print(pro1.shape)
dice, jac = dice_coeff((pro1>=0.5).astype('uint8') * 255, target)
print("pro1, dice is:", dice, "jac is:", jac)
pro2 = load_nifty_volume_as_array(flip_2_path)
# 165  *  512
pro2 = np.transpose(pro2, [1,0,2])
print(pro2.shape)
dice, jac = dice_coeff((pro2>=0.5).astype('uint8') * 255, target)
print("pro2, dice is:", dice, "jac is:", jac)
pro3 = load_nifty_volume_as_array(flip_3_path)
pro3 = np.transpose(pro3, [1,2,0])
print(pro3.shape)
dice, jac = dice_coeff((pro3 >= 0.5).astype('uint8') * 255, target)
print("pro3, dice is:", dice, "jac is:", jac)

desired_path =  './history/'
pro = pro1 + pro2 + pro3
pro = pro / 3
export_name = 'aug_test_pro' + '.nii.gz'
save_array_as_nii_volume(pro, desired_path + export_name)
export_name = 'aug_test' + '.nii.gz'
pred = pro >= 0.5
pred = pred.astype('uint8')
save_array_as_nii_volume(pred, desired_path + export_name)

pred[pred == 1] = 255
print(np.unique(pred))

dice, jac = dice_coeff(pred, target)
print("dice is:", dice, "jac is:", jac)
save_path = './com.nii.gz'
save_com_img2nii(pred, target, save_path)


