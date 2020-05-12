import numpy as np
from PIL import Image
import glob
import torch, math,  random
import torch.nn as nn
from torch.autograd import Variable
from random import randint
from torch.utils.data.dataset import Dataset
from pre_processing import *
from mean_std import *
from util.data_aug import *
from util.tools_self import cal_crop_num, load_nifty_volume_as_array,save_array_as_nii_volume
import scipy.misc
from skimage import transform
Training_MEAN = 141.06229505729505
Training_STDEV = 28.51522462173067

#
# def sort_slices(msk, img):
#     data_len = msk.shape[0]
#     easy_slices = []
#     eh_slices = []
#     hard_slices = []
#     for i in range(data_len):
#         if np.sum(msk[i, ...]) == 0:
#             hard_slices.append(msk[i, ...])
#             continue
#         if np.sum(msk_datasets[msk_slices[i]]) <= 500:
#             eh_slices.append(msk_slices[i])
#             continue
#         easy_slices.append(msk_slices[i])
#     return easy_slices, eh_slices, hard_slices


class SEMDataTrain(Dataset):

    def __init__(self, image_path, mask_path, reg_path, in_size=572, out_size=388):
        """
           :param image_path: path of raw image,  img shape is [z, y ,x] [165, 768, 1024]
           :param mask_path: path of GT,  GT shape is [z, y ,x] [165, 768, 1024]
           :param in_size: [x, y, z] of patch size send to Net
           :param out_size: [x, y, z] of patch size out of net
        """

        # load img
        self.img = load_nifty_volume_as_array(image_path)
        self.img = self.img.astype('float')
        self.img_depth, self.img_height, self.img_width = self.img.shape[0], self.img.shape[1], self.img.shape[2]

        self.lab = load_nifty_volume_as_array(mask_path)
        self.lab = approximate_image(self.lab)  # images only with 0 and 255

        # jieduan = math.floor(self.img.shape[0] * 0.9)
        # self.img = self.img[:jieduan, ...]
        # self.lab = self.lab[:jieduan, ...]

        pdsz = 30
        self.img = np.pad(self.img, pdsz, mode='symmetric')
        self.lab = np.pad(self.lab, pdsz, mode='symmetric')

        # Normalize the image
        self.ori_img = self.img
        self.img_mean, self.img_std = np.mean(self.img), np.std(self.img)

        crop_n1, crop_n2, crop_n3 = cal_crop_num(self.img.shape, in_size)
        self.img = multi_cropping(self.img,crop_size=in_size,
                                  crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)
        # save_array_as_nii_volume(self.img[20,...],'./t1.nii.gz')
        assert crop_n1 * crop_n2 * crop_n3 == self.img.shape[0], "false  crop in training."
        # load lab

        self.ori_lab = self.lab
        self.lab = multi_cropping(self.lab, crop_size=out_size, crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)

        # load reg
        self.reg = load_nifty_volume_as_array(reg_path)
        self.reg = self.reg.astype('float')
        self.ori_reg = self.reg
        self.reg = multi_cropping(self.reg, crop_size=out_size, crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)

        self.reg_zy = load_nifty_volume_as_array('oriCvLab/train_yz_proximity.nii.gz')
        self.reg_zy = self.reg_zy.astype('float')
        self.reg_zx = load_nifty_volume_as_array('oriCvLab/train_xz_proximity.nii.gz')
        self.reg_zx = self.reg_zx.astype('float')

        self.data_len = self.img.shape[0]
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        img_as_np = self.img[index, ...]
        msk_as_np = self.lab[index, ...]
        reg_as_np = self.reg[index, ...]
        # random choose a patch from the volume
        if random.random() > 0.7:
            count_sample = 0
            while count_sample < 50:
                count_sample += 1
                z_loc, y_loc, x_loc = randint(0, self.img_depth - self.in_size[0]), \
                                      randint(0, self.img_height - self.in_size[1]), \
                                      randint(0, self.img_width - self.in_size[2])

                img_as_np = cropping(self.ori_img, crop_size=self.in_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                    'float')
                # img_as_np = normalization2(img_as_np, max=1, min=0)
                # Crop the mask
                msk_as_np = cropping(self.ori_lab, crop_size=self.in_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                    'uint8')
                # Crop the reg
                reg_as_np = cropping(self.ori_reg, crop_size=self.in_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                    'float')
                msk_as_np_one = msk_as_np // 255
                # img_as_np = normalization2(img_as_np, max=1, min=0)
                if msk_as_np_one.sum() > (self.in_size[0] * self.in_size[1] * self.in_size[2]) // 80:
                    break

        # resize a patch choose from the volume
        if random.random() > 0.9:
            resize_size = [self.in_size[0], randint(self.in_size[1], 368), randint(self.in_size[2], 512)]
            # resize_size = [self.in_size[0], randint(self.in_size[1], self.img_height),
            # randint(self.in_size[2], self.img_width)]
            z_loc, y_loc, x_loc = randint(0, self.img_depth - resize_size[0]), \
                                  randint(0, self.img_height - resize_size[1]), \
                                  randint(0, self.img_width - resize_size[2])

            img_as_np = cropping(self.ori_img, crop_size=resize_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                'float')
            # Crop the mask
            msk_as_np = cropping(self.ori_lab, crop_size=resize_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                'uint8')

            # Crop the reg
            reg_as_np = cropping(self.ori_reg, crop_size=resize_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                'float')

            img_as_np = transform.resize(img_as_np, self.in_size, preserve_range=True, mode='constant')
            msk_as_np = transform.resize(msk_as_np, self.in_size, preserve_range=True, mode='constant')
            msk_as_np = approximate_image(msk_as_np).astype('uint8')  # images only with 0 and 255
            reg_as_np = transform.resize(reg_as_np, self.in_size, preserve_range=True, mode='constant')

            # print('nuique msk', np.unique(msk_as_np))
            # assert np.max(img_as_np) <= 1 and np.min(img_as_np) >= 0, 'img not '
            if len(np.unique(msk_as_np)) == 1:
                assert np.max(msk_as_np) == 255 or np.min(msk_as_np) == 0, 'mask not {0,255}'
            if len(np.unique(msk_as_np)) == 2:
                assert np.max(msk_as_np) == 255 and np.min(msk_as_np) == 0, 'mask not {0,255}'
            assert img_as_np.shape == tuple(self.in_size), 'resize is not true'
            # msk_as_np = approximate_image(msk_as_np)
            # filename = './trainaug.nii.gz'
            # save_array_as_nii_volume(img_as_np, filename)
            # filename = './trainauglab.nii.gz'
            # save_array_as_nii_volume(msk_as_np, filename)

        if random.random() > 0.7:
            if random.random() > 0.5:
                tpimg = np.transpose(self.ori_img, [1, 0, 2])
                tplab = np.transpose(self.ori_lab, [1, 0, 2])
                tpreg = self.reg_zx
            else:
                tpimg = np.transpose(self.ori_img, [2, 0, 1])
                tplab = np.transpose(self.ori_lab, [2, 0, 1])
                tpreg = self.reg_zy

            z_loc, y_loc, x_loc = randint(0, tpimg.shape[0] - self.in_size[0]), \
                                  randint(0, tpimg.shape[1] - self.in_size[1]), \
                                  randint(0, tpimg.shape[2] - self.in_size[2])

            img_as_np = cropping(tpimg, crop_size=self.in_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                'float')
            # Crop the mask
            msk_as_np = cropping(tplab, crop_size=self.in_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                'uint8')
            reg_as_np = cropping(tpreg, crop_size=self.in_size, dim1=z_loc, dim2=y_loc, dim3=x_loc).astype(
                'float')

        if random.random() > 0.4:
            img_as_np, msk_as_np, reg_as_np = aug_img_lab_reg(np.transpose(img_as_np, [1, 2, 0]),
                                                   np.transpose(msk_as_np, [1, 2, 0]),
                                                   np.transpose(reg_as_np, [1, 2, 0]),1)
            msk_as_np = approximate_image(msk_as_np)  # images only with 0 and 255

        assert np.min(msk_as_np) == 0 or np.max(msk_as_np) == 255 or \
               np.min(msk_as_np) == 0 and np.max(msk_as_np) == 255, 'label is not {0, 255}'

        img_as_np = (img_as_np - self.img_mean) / self.img_std
        # filename = './trainaug.nii.gz'
        # save_array_as_nii_volume(img_as_np, filename)
        img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
        img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np / 255
        reg_as_np = reg_as_np / 255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        # msk_r_as_np = cal_dist2lab(msk_as_np)
        msk_r_as_tensor = torch.from_numpy(reg_as_np).float()  # Convert numpy array to tensor

        return img_as_tensor, msk_as_tensor, msk_r_as_tensor

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.img.shape[0]


class SEMDataVal(Dataset):
    def __init__(self, image_path, mask_path, in_size, out_size):
        '''
        Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # load img and lab
        self.img = load_nifty_volume_as_array(image_path)
        self.img = self.img.astype('float')
        # Normalize the image
        # self.img = normalization2(self.img, max=1, min=0)
        self.img_mean, self.img_std = np.mean(self.img), np.std(self.img)
        self.lab = load_nifty_volume_as_array(mask_path)

        pdsz = 20
        self.img = np.pad(self.img, pdsz, mode='symmetric')
        self.lab = np.pad(self.lab, pdsz, mode='symmetric')

        self.lab = approximate_image(self.lab)  # images only with 0 and 255

        self.data_len = 1
        self.img_depth, self.img_height, self.img_width = self.img.shape[0], self.img.shape[1], self.img.shape[2]
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index : an integer variable that calls (indext)th image in the
                    path
        Returns:
            Tensor: 4 cropped data on index which is converted to Tensor
        """
        img_size = self.img.shape

        crop_n1, crop_n2, crop_n3 = cal_crop_num(img_size, self.in_size)
        img_as_np = multi_cropping(self.img,
                                   crop_size=self.in_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)

        # Empty list that will be filled in with arrays converted to tensor
        processed_list = []

        for array in img_as_np:
            # Normalize the cropped arrays
            # img_to_add = normalization2(array, max=1, min=0)
            img_to_add = (array - self.img_mean) / self.img_std
            processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(processed_list)
        #  return tensor of 4 cropped images
        #  top left, top right, bottom left, bottom right respectively.

        """
        # GET MASK
        """
        # Normalize mask to only 0 and 1
        msk_as_np = multi_cropping(self.lab, crop_size=self.out_size, crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)

        msk_as_np = msk_as_np / 255

        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor

        return img_as_tensor, msk_as_tensor, self.lab

    def __len__(self):

        return self.data_len


class SEMDataTest(Dataset):
    def __init__(self, image, msk, in_size, out_size):
        # load img and lab
        self.img = image
        self.img = self.img.astype('float')
        # Normalize the image
        self.img_mean, self.img_std = np.mean(self.img), np.std(self.img)
        self.lab = msk
        self.lab = approximate_image(self.lab)  # images only with 0 and 255

        pdsz = 20
        self.img = np.pad(self.img, pdsz, mode='symmetric')
        self.lab = np.pad(self.lab, pdsz, mode='symmetric')

        self.data_len = 1
        self.img_depth, self.img_height, self.img_width = self.img.shape[0], self.img.shape[1], self.img.shape[2]
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        img_size = self.img.shape

        crop_n1, crop_n2, crop_n3 = cal_crop_num(img_size, self.in_size)
        # print('crop  num is ', crop_n1, crop_n2,  crop_n3)
        img_as_np = multi_cropping(self.img,
                                   crop_size=self.in_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)

        processed_list = []
        for array in img_as_np:
            # Normalize the cropped arrays
            img_to_add = (array - self.img_mean) / self.img_std
            processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(processed_list)
        msk_as_np = multi_cropping(self.lab,
                                   crop_size=self.out_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)
        msk_as_np = msk_as_np / 255

        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        original_msk = torch.from_numpy(msk_as_np)
        return img_as_tensor, msk_as_tensor, self.lab

    def __len__(self):
        return self.data_len


class SEMDataVal_hard(Dataset):
    def __init__(self, image_path, mask_path, in_size, out_size):
        ''' Args:
            image_path = path where test images are located
            mask_path = path where test masks are located
        '''
        # load img and lab
        hard_img_path = './oriCvLab/valnii/img/'
        hard_lab_path = './oriCvLab/valnii/lab/'
        self.img_mean = Training_MEAN
        self.img_std = Training_STDEV

        self.img_list = glob.glob(hard_img_path + '*.nii.gz')
        self.lab_list = glob.glob(hard_lab_path + '*.nii.gz')
        self.img_list.sort()
        self.lab_list.sort()
        self.data_len = len(self.img_list)
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        single_vol = self.img_list[index]
        img_numpy = load_nifty_volume_as_array(single_vol)

        single_vol_lab = self.lab_list[index]
        lab_numpy = load_nifty_volume_as_array(single_vol_lab)
        lab_numpy = approximate_image(lab_numpy)
        ori_lab = lab_numpy

        pdsz = 20
        img_numpy = np.pad(img_numpy, pdsz, mode='symmetric')
        ori_lab = np.pad(ori_lab, pdsz, mode='symmetric')

        img_size = img_numpy.shape
        crop_n1, crop_n2, crop_n3 = cal_crop_num(img_size, self.in_size)
        # print('crop  num is ', crop_n1, crop_n2,  crop_n3)
        img_as_np = multi_cropping(img_numpy, crop_size=self.in_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)
        # Empty list that will be filled in with arrays converted to tensor
        processed_list = []
        for array in img_as_np:
            # Normalize the cropped arrays
            img_to_add = (array - self.img_mean) / self.img_std
            processed_list.append(img_to_add)

        img_as_tensor = torch.Tensor(processed_list)
        #  return tensor of 4 cropped images
        #  top left, top right, bottom left, bottom right respectively.
        # GET MASk
        msk_as_np = multi_cropping(lab_numpy, crop_size=self.out_size,
                                   crop_num1=crop_n1, crop_num2=crop_n2, crop_num3=crop_n3)
        msk_as_np = msk_as_np / 255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        original_msk = torch.from_numpy(msk_as_np)
        return img_as_tensor, msk_as_tensor, ori_lab

    def __len__(self):
        return self.data_len


if __name__ == "__main__":

    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')
    SEM_val = SEMDataVal('../data/val/images', '../data/val/masks')

    imag_1, msk = SEM_train.__getitem__(0)
