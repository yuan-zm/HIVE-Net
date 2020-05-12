from advanced_model import UNet_3D
from modules import *
from save_history import *
from losses import *
from util.tools_self import find_last
import time
from configFile.add_argument import get_arguments


def train():

    args = get_arguments()
    print(args)
    # Dataloader end
    train_load, val_load, val_hard_load = parpare_dataset(args)
    # Model
    model = UNet_3D(in_dim=args.in_channels, out_dim=args.out_channels, num_filters=args.init_ft_num)
    # assign gpu devices
    if args.useallgpu:
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpuDevice)

    # Loss function
    # criterion = nn.CrossEntropyLoss()
    # criterion = dice_loss()
    criterion = jacc_loss()
    criterion2 = nn.MSELoss()   # extend_mse_loss(beta=0.2, lamda=1)
    # Optimizerd
    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    # Parameters
    epoch_start = args.start_epoch
    epoch_end = args.end_epoch
    iter_start = args.start_iter
    tra_iter = iter_start

    if args.start_epoch != 0 and iter_start != 0:
        print('epoch_start and iter_start both non-zero')
        exit()
    if args.start_epoch != 0:
        model = load_old_model(args, model)
    if args.start_iter != 0:
        model = load_old_model(args, model)

    # Saving History to csv
    header = ['epoch', 'dice', 'jac', 'train loss', 'train acc', 'val loss', 'val acc']
    header_val_hard = ['epoch', 'dice', 'jac', 'val loss', 'val acc']
    save_file_name = args.val_seg_csv_path
    folder_index = find_last(save_file_name, '/')
    save_dir = save_file_name[:folder_index]

    hard_save_file_name = args.hard_val_seg_csv_path
    folder_index = find_last(hard_save_file_name, '/')
    hard_save_dir = hard_save_file_name[:folder_index]
    # Saving images and models directories
    model_save_dir = args.model_save_path

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):

        start_time = time.time()
        # train the model
        model.train()
        for batch, (images, masks, masks_r) in enumerate(train_load):
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            masks_r = Variable(masks_r.cuda())
            outputs, outputs_r = model(images)
            loss = criterion(outputs, masks)
            outputs_r = torch.squeeze(outputs_r, dim=1)
            loss2 = criterion2(outputs_r, masks_r)
            all_loss = 0.7 * loss + 0.3 * loss2
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            tra_iter += 1
            if tra_iter % 10 == 0:
                print('In epoch {}, iteration is:{}, loss seg is:{}, loss reg is:{}'.format(i+1, tra_iter, loss, loss2))

            if tra_iter % 200 == 0:
                start_hard_time = time.time()
                val_acc, val_loss, dice, jac_iter = validate_hard_model(model, val_hard_load, criterion,
                                                                        tra_iter, True, args.itera_image_save_path)
                end_hard_time = time.time()
                print('this is {} iteration, dice is:{:.6}, '
                      'jac is:{:.6}, val_acc is:{:.6}, val_loss is:{:.6}, val hard is:{:.6}'
                      .format(tra_iter, dice, jac_iter, val_acc, val_loss, end_hard_time-start_hard_time))
                values = [tra_iter, dice, jac_iter, val_loss, val_acc]
                export_history(header_val_hard, values, hard_save_dir, hard_save_file_name)

                save_models(model, args.itera_model_save_prefix, tra_iter)

        train_acc, train_loss = get_loss_train(model, train_load, criterion)

        end_time = time.time()

        if scheduler is not None:
            scheduler.step(epoch=i)
        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Time:", end_time - start_time,
              "learning rate:", scheduler.get_lr()[0], "Train acc", train_acc)

        # Validation every val_epoch epoch
        if (i+1) % args.val_epoch == 0:
            val_acc, val_loss, dice, jac = validate_model(
                model, val_load, criterion, i+1, True, args.image_save_path)
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, dice, jac, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i+1) % args.snapshot_epoch == 0:  # save model every 10 epoch
                save_models(model, model_save_dir, i+1)


def load_old_model(args, model):
    if args.start_epoch != 0 and args.start_iter != 0:
        print('epoch_start and iter_start both non-zero')
        exit()
    if args.start_epoch != 0:
        model_save_prefix = args.epoch_model_save_prefix
        start_point = args.start_epoch
    if args.start_iter != 0:
        model_save_prefix = args.itera_model_save_prefix
        start_point = args.start_iter

    model_path = model_save_prefix + str(start_point) + ".pwf"
    old_model = torch.load(model_path)
    model_dict = model.state_dict()

    pretrained_dict = old_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def parpare_dataset(args):
    # Dataset start
    tra_img_path = args.train_image_path
    tra_lab_path = args.train_label_path
    tra_reg_path = args.train_reg_path
    in_size = args.in_size
    out_size = args.out_size
    val_img_path = args.val_image_path
    val_lab_path = args.val_label_path
    batch_size = args.batch_size
    dataset_train = SEMDataTrain(image_path=tra_img_path, mask_path=tra_lab_path, reg_path=tra_reg_path, in_size=in_size, out_size=out_size)
    dataset_val = SEMDataVal(image_path=val_img_path, mask_path=val_lab_path,in_size=in_size, out_size=out_size)
    SEM_val_hard = SEMDataVal_hard(image_path=val_img_path, mask_path=val_lab_path, in_size=in_size, out_size=out_size)

    val_hard_load = torch.utils.data.DataLoader(dataset=SEM_val_hard, num_workers=3, batch_size=1, shuffle=False)
    train_load = torch.utils.data.DataLoader(dataset=dataset_train,num_workers=40, batch_size=batch_size, shuffle=True)
    val_load = torch.utils.data.DataLoader(dataset=dataset_val,num_workers=3, batch_size=1, shuffle=False)
    return train_load, val_load, val_hard_load


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print('Number of argss should be 2. e.g.')
    #     print('    python main.py ./configFile/trainConfig.txt')
    #     exit()
    # config_file = str(sys.argv[1])
    # assert(os.path.isfile(config_file))
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()
    print('data 0 mean 1 std'
          'prelu'
          'aug transpose')

