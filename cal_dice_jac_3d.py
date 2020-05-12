import nibabel, glob
import numpy as np
from util.tools_self import save_array_as_nii_volume
from PIL import Image
from metrics import *


def save_img2nii(img_path, save_path):

    img_arr = glob.glob(str(img_path) + str("/*"))
    img_arr.sort()
    single_image_name = img_arr[0]
    img_as_img = Image.open(single_image_name)
    img_as_np = np.asarray(img_as_img).astype('uint8')
    all_lab = np.zeros((img_as_np.shape[0], img_as_np.shape[1], len(img_arr)))

    for i in range(len(img_arr)):
        single_image_name = img_arr[i]
        img_as_img = Image.open(single_image_name)
        img_as_np = np.asarray(img_as_img).astype('uint8')
        all_lab[:,:,i] = img_as_np


    # img_as_img.show()

    save_array_as_nii_volume(np.transpose(all_lab, [2,0,1]), save_path)


def save_com_img2nii(pred, lab, save_path):

    lab[lab == 255] = 5
    pred[pred == 255] = 1
    re = lab - pred
    re[re == 255] = 3
    re[re == 4] = 2
    re[re == 5] = 1

    print(np.unique(re[:]))
    save_array_as_nii_volume(re, save_path)


def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data


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


if __name__ == '__main__':
    test_epoch = 175
    pred_path = './history/RMS/result_images3/epoch_' + str(test_epoch) + '/' + str(test_epoch) + '.nii.gz'
    pred_path = './history/aug_test.nii.gz'
    # pred_path = '375.nii.gz'
    # pred_path = './history/RMS/result_images3/epoch_' + str(test_epoch) + '/'+ str(test_epoch) + '.nii.gz'
    lab_path = './oriCvLab/testLab.nii.gz'
    pred = load_nifty_volume_as_array(pred_path) *255
    pred = approximate_image(pred)

    target = load_nifty_volume_as_array(lab_path)
    target = approximate_image(target)
    dice, jac = dice_coeff(pred, target)
    print("dice is:", dice, "jac is:", jac)
    save_path = './com.nii.gz'
    save_com_img2nii(pred, target, save_path)

# if __name__ == "__main__":
#     lab_path = '/media/peng/D/yzm/adjslice/oriCvLab/testCvlab/lab'
#     save_path = './test_lab.nii.gz'
#     save_img2nii(lab_path, save_path)
