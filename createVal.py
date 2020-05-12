import SimpleITK as sitk
import numpy as np
import nibabel
from util.tools_self import *


def cropping(image, crop_size, dim1, dim2, dim3):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[dim1:dim1+crop_size[0], dim2:dim2+crop_size[1], dim3:dim3+crop_size[2]]
    return cropped_img

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
    data = np.transpose(data, [2, 1, 0])
    if with_header:
        return data, img.affine, img.header
    else:
        return data

def save_array_as_nii_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    # data = np.flipud(data)
    # data = np.fliplr(data)
    # data =  np.transpose(data, [2, 0, 1])
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def stride_size(image_len, crop_num, crop_size):
    return int((image_len - crop_size)/(crop_num - 1))

def multi_cropping(image, crop_size, crop_num1, crop_num2, crop_num3):
    """crop the image and pad it to in_size
    Args :
        images : numpy arrays of images
        crop_size(int) : size of cropped image
        crop_num2 (int) : number of crop in horizontal way
        crop_num1 (int) : number of crop in vertical way
    Return :
        cropped_imgs : numpy arrays of stacked images
    """

    img_depth, img_height, img_width = image.shape[0], image.shape[1], image.shape[2]

    # assert crop_size * crop_num1 >= img_height and crop_size * \
    #        crop_num2 >= img_width, "Whole image cannot be sufficiently expressed"

    assert crop_size[1]*crop_num2 >= img_height  and crop_size[2] * \
        crop_num3 >= img_width, "Whole image cannot be sufficiently expressed"
    # assert crop_num1 <= img_width - crop_size + 1 and crop_num2 <= img_height - \
    #     crop_size + 1, "Too many number of crops"

    cropped_imgs = []
    # int((img_height - crop_size)/(crop_num1 - 1))
    dim1_stride = stride_size(img_depth, crop_num1, crop_size[0])
    # int((img_width - crop_size)/(crop_num2 - 1))
    dim2_stride = stride_size(img_height, crop_num2, crop_size[1])
    dim3_stride = stride_size(img_width, crop_num3, crop_size[2])
    for i in range(crop_num1):
        for j in range(crop_num2):
            for k in range(crop_num3):
                cropped_imgs.append(cropping(image, crop_size,
                                         dim1_stride*i, dim2_stride*j, dim3_stride*k))
    return np.asarray(cropped_imgs)


img = load_nifty_volume_as_array('./oriCvLab/testImg.nii.gz')
msk = load_nifty_volume_as_array('./oriCvLab/testLab.nii.gz')
in_size = [80, 368, 512]

crop_n1, crop_n2, crop_n3 = cal_crop_num(img.shape, in_size)
img = multi_cropping(img, crop_size=in_size,crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)
assert crop_n1 * crop_n2 * crop_n3 == img.shape[0], "false  crop in training."
msk = multi_cropping(msk, crop_size=in_size,crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)

fold_name = './split_test_data'
for i in range(img.shape[0]):
    fn = fold_name + '/img/' + 'test_' + str(i) + '.nii.gz'
    save_array_as_nii_volume(img[i, ...], fn)
    fn = fold_name + '/msk/' + 'test_lab_' + str(i) + '.nii.gz'
    save_array_as_nii_volume(msk[i, ...], fn)
