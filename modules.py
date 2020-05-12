import numpy as np
from PIL import Image
import torch
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


def train_model(model, data_train, criterion, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    model.train()
    for batch, (images, masks) in enumerate(data_train):
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)
        # print(masks.shape, outputs.shape)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        # Update weights
        optimizer.step()
    # total_loss = get_loss_train(model, data_train, criterion)


def get_loss_train(model, data_train, criterion):
    """
        Calculate loss over train set
    """
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (images, masks, masks_r) in enumerate(data_train):
        with torch.no_grad():
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            outputs, outputs_r = model(images)

            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1).float()
            acc = accuracy_check_for_batch(masks.cpu(), preds.cpu(), images.size()[0])
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
    return total_acc / (batch + 1), total_loss / (batch + 1)


def validate_model(model, data_val, criterion, epoch, make_prediction=True, save_folder_name='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    for batch, (images_v, masks_v, original_msk) in enumerate(data_val):
        pdsz = 20
        ori_shape = original_msk.shape
        original_msk = original_msk[...,pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
        stacked_img = torch.Tensor([]).cuda()
        # stacked_dis = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda())
                mask_v = Variable(masks_v[:, index, :, :].squeeze(1).cuda())
                # print(image_v.shape, mask_v.shape)
                output_v, output_r_v = model(image_v)
                total_val_loss = total_val_loss + criterion(output_v, mask_v).cpu().item()
                # print('out', output_v.shape)
                output_v = torch.argmax(output_v, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_v))
                # stacked_dis = torch.cat((stacked_dis, output_r_v))
                # output_r_v = torch.squeeze(output_r_v)

        if make_prediction:
            im_name = batch  # TODO: Change this to real image name so we know
            # reconstruct_image(stacked_dis, epoch, save_folder_name)
            pred_msk = save_prediction_image(stacked_img, ori_shape[-3:], im_name, epoch, 0, save_folder_name)
            acc_val = accuracy_check(original_msk, pred_msk)
            dice, jac = dice_coeff(pred_msk, original_msk)
            total_val_acc = total_val_acc + acc_val
            print('val dice:{0}, jac:{1} in epoch{2}'.format(dice, jac, epoch))

    return total_val_acc/(batch + 1), total_val_loss/(+(batch + 1)*4), dice, jac


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
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda())
                output_v = model(image_v)
                output_v = torch.argmax(output_v, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_v))

        im_name = batch  # TODO: Change this to real image name so we know
        pred_msk = save_prediction_image(stacked_img, ori_shape[-3:], im_name, epoch, 0, save_folder_name)
        acc_val = accuracy_check(original_msk, pred_msk)
        avg_dice, jac = dice_coeff(pred_msk, original_msk)
        total_val_jac += jac
        total_val_dice += avg_dice
        total_val_acc = total_val_acc + acc_val

    print("total_val_acc is:%f.  total_val_jac is:%f . total_val_dice is:%f "
          "Finish Prediction!" % (total_val_acc / (batch + 1), total_val_jac / (batch + 1), total_val_dice / (batch + 1)))
    header = ['epoch', 'total_val_jac', 'total_val_dice', 'total_val_acc']
    values = [epoch, total_val_jac/ (batch + 1), total_val_dice/ (batch + 1), total_val_acc/ (batch + 1)]

    export_history(header, values, save_dir, save_file_name)
    return total_val_acc / (batch + 1)


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
    # new_img_size = [ori_shape[0] + 2*pdsz, ori_shape[1] + 2*pdsz, ori_shape[2] + 2*pdsz]

    crop_n1, crop_n2, crop_n3 = cal_crop_num(ori_shape, in_size)

    div_arr = division_array(stacked_img_shape[1:], crop_n1, crop_n2, crop_n3, ori_shape) + 0.000001
    img_cont_np = image_concatenate(stacked_img, crop_n1, crop_n2,
                                    crop_n3,ori_shape[0] , ori_shape[1] , ori_shape[2])

    probability = img_cont_np / div_arr
    # print(np.unique(probability))
    probability = probability[pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
    if indate_it == 0:
        desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    else:
        desired_path = save_folder_name + '/iter_' + str(epoch) + '/'

    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(epoch) + '_pro.nii.gz'
    save_array_as_nii_volume(probability, desired_path + export_name)

    img_cont_np = img_cont_np.astype('uint8')
    #img_cont_np = img_cont_np[pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
    img_cont_np = polarize((img_cont_np)/div_arr)*255
    img_cont_np = img_cont_np[..., pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
    # Save Image!
    export_name = str(epoch) + '.nii.gz'
    save_array_as_nii_volume(img_cont_np, desired_path+ export_name)
    return img_cont_np


def validate_hard_model(model, data_val, criterion, epoch, make_prediction=True, save_folder_name='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    total_val_dice = 0
    total_val_jac = 0
    count_val = 0
    for batch, (images_v, masks_v, original_msk) in enumerate(data_val):
        pdsz = 20
        ori_shape = original_msk.shape
        original_msk = original_msk[...,pdsz:-pdsz, pdsz:-pdsz, pdsz:-pdsz]
        stacked_img = torch.Tensor([]).cuda()
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda())
                mask_v = Variable(masks_v[:, index, :, :].squeeze(1).cuda())
                # print(image_v.shape, mask_v.shape)
                output_v, output_r_v = model(image_v)
                total_val_loss = total_val_loss + criterion(output_v, mask_v).cpu().item()
                # print('out', output_v.shape)
                output_v = torch.argmax(output_v, dim=1).float()
                stacked_img = torch.cat((stacked_img, output_v))
        if make_prediction:
            im_name = batch  # TODO: Change this to real image name so we know
            pred_msk = save_prediction_image(stacked_img, ori_shape[-3:], im_name, epoch, 1, save_folder_name)
            acc_val = accuracy_check(original_msk, pred_msk)
            dice, jac = dice_coeff(pred_msk, original_msk)
            total_val_acc = total_val_acc + acc_val
            total_val_dice = total_val_dice + dice
            total_val_jac = total_val_jac + jac

        count_val += 1

    return total_val_acc/(count_val + 1), total_val_loss/((count_val + 1)*4),\
           total_val_dice/(count_val + 1), total_val_jac/(count_val + 1)


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


if __name__ == '__main__':
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    SEM_train_load = torch.utils.data.DataLoader(dataset=SEM_train,
                                                 num_workers=3, batch_size=10, shuffle=True)
    get_loss_train()
