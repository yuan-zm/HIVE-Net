
import torch
import os
from dataset import SEMDataTest
from util.tools_self import load_nifty_volume_as_array,save_array_as_nii_volume
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from dataset import *
import torch.nn as nn
from accuracy import accuracy_check, accuracy_check_for_batch
import csv
import os
from metrics import *
from save_history import export_history
from util.tools_self import save_array_as_nii_volume




def test_model(model_path, data_test, epoch, save_folder_name='prediction',
               save_dir="./history/RMS/result_images_test",
               save_file_name = "./history/RMS/result_images_test/history_RMS3.csv"):
    """
        Test run
    """
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
    model.eval()

    total_val_acc = 0
    total_val_jac = 0
    total_val_dice = 0
    for batch, (images_v, masks_v, original_msk) in enumerate(data_test):
        pdsz = 20
        ori_shape = original_msk.shape
        original_msk = original_msk[...,pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
        stacked_img = torch.Tensor([]).cuda()
        stacked_reg = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda())
                output_v, output_r_v = model(image_v)
                output_v = torch.argmax(output_v, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_v))
                output_r_v = torch.squeeze(output_r_v, dim=0)
                stacked_reg = torch.cat((stacked_reg, output_r_v))

        im_name = batch  # TODO: Change this to real image name so we know
        pred_msk = save_prediction_image(stacked_img, ori_shape[-3:], im_name, epoch, 0, save_folder_name)
        acc_val = accuracy_check(original_msk, pred_msk)
        avg_dice, jac = dice_coeff(pred_msk, original_msk)
        total_val_jac += jac
        total_val_dice += avg_dice
        total_val_acc = total_val_acc + acc_val

        reconstruct_image(stacked_reg, ori_shape[-3:], epoch, save_folder_name)

    print("total_val_acc is:%f.  total_val_jac is:%f . total_val_dice is:%f "
          "Finish Prediction!" % (total_val_acc / (batch + 1), total_val_jac / (batch + 1), total_val_dice / (batch + 1)))
    header = ['epoch', 'total_val_jac', 'total_val_dice', 'total_val_acc']
    values = [epoch, total_val_jac/ (batch + 1), total_val_dice/ (batch + 1), total_val_acc/ (batch + 1)]

    export_history(header, values, save_dir, save_file_name)
    return total_val_acc / (batch + 1)


def reconstruct_image(stacked_img, ori_shape, epoch, save_folder_name):
    stacked_img = stacked_img.cpu().data.numpy()
    stacked_img_shape = np.shape(stacked_img)
    in_size = np.shape(stacked_img)[1:]
    pdsz = 20

    crop_n1, crop_n2, crop_n3 = cal_crop_num(ori_shape, in_size)
    div_arr = division_array(stacked_img_shape[1:], crop_n1, crop_n2, crop_n3, ori_shape) + 0.000001

    img_cont_np = image_concatenate(stacked_img, crop_n1, crop_n2,
                                    crop_n3, ori_shape[0], ori_shape[1], ori_shape[2])
    img_cont = img_cont_np / div_arr
    img_cont_np = img_cont.astype('float')
    # pdsz = 30
    img_cont_np = img_cont_np[pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(epoch) + '_dis.nii.gz'
    save_array_as_nii_volume(img_cont_np, desired_path + export_name)

    return img_cont


def save_prediction_image(stacked_img, ori_shape, im_name, epoch, indate_it=0, save_folder_name="result_images", save_im=True):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
        division_array(388, 2, 3, 768, 1024):
    """
    stacked_img = stacked_img.cpu().data.numpy()
    stacked_img_shape = np.shape(stacked_img)
    in_size = np.shape(stacked_img)[1:]
    stacked_size = stacked_img_shape[0]
    pdsz = 20

    crop_n1, crop_n2, crop_n3 = cal_crop_num(ori_shape, in_size)

    div_arr = division_array(stacked_img_shape[1:], crop_n1, crop_n2, crop_n3, ori_shape) + 0.000001
    img_cont_np = image_concatenate(stacked_img, crop_n1, crop_n2,
                                    crop_n3, ori_shape[0], ori_shape[1], ori_shape[2])

    probability = img_cont_np / div_arr
    # print(np.unique(probability))
    probability = probability[pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(epoch) + '_pro.nii.gz'
    save_array_as_nii_volume(probability, desired_path + export_name)

    img_cont_np = img_cont_np.astype('uint8')
    img_cont_np = polarize((img_cont_np)/div_arr)*255
    img_cont_np = img_cont_np[..., pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
    # organize images in every epoch
    if indate_it == 0:
        desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    else:
        desired_path = save_folder_name + '/iter_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(epoch) + '.nii.gz'
    save_array_as_nii_volume(img_cont_np, desired_path+ export_name)

    return img_cont_np


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_epoch = xxx
test_iter = 0

image_path = './oriCvLab/testImg.nii.gz'
label_path = './oriCvLab/testLab.nii.gz'

for test_aug in range(3):
#test_aug = 2

    if test_epoch != 0:
        model_name = "./history/RMS/saved_models3/model_epoch_" + str(test_epoch) + ".pwf"
    else:
        model_name = "./history/RMS/itera_saved_models3/model_itera_/model_epoch_" + str(test_iter) + ".pwf"
    img_save_path = "./history/RMS/result_images_test"

    in_size = [40, 136, 136]
    out_size = in_size

    img = load_nifty_volume_as_array(image_path)
    lab = load_nifty_volume_as_array(label_path)

    if test_aug == 1:
        img = np.transpose(img, [1, 0, 2])
        lab = np.transpose(lab, [1, 0, 2])
        img_save_path = "./history/RMS/result_images_test_1"
    if test_aug == 2:
        img = np.transpose(img, [2, 0, 1])
        lab = np.transpose(lab, [2, 0, 1])
        img_save_path = "./history/RMS/result_images_test_2"

    SEM_test = SEMDataTest(img, lab, in_size, out_size)

    SEM_test_load = torch.utils.data.DataLoader(dataset=SEM_test, num_workers=3, batch_size=1, shuffle=False)

    print("generate test prediction")
    test_model(model_name, SEM_test_load, test_epoch, img_save_path, save_file_name="./history/RMS/result_images_test/history_RMS3.csv")
