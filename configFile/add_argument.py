import argparse
# [DATA]
DATASET_NAME = "CVLAB"
TRAIN_IMAGE_PATH = "./oriCvLab/trainImg.nii.gz"
TRAIN_LABEL_PATH = "./oriCvLab/trainLab.nii.gz"
TRAIN_REG_PATH = "./oriCvLab/train_proximity_pad.nii.gz"
VAL_IMAGE_PATH = "./oriCvLab/testImg.nii.gz"
VAL_LABEL_PATH = "./oriCvLab/testLab.nii.gz"
IN_SIZE_X = 186
IN_SIZE_Y = 186
IN_SIZE_Z = 20
IN_SIZE = [40, 136, 136]
OUT_SIZE_X = 186
OUT_SIZE_Y = 186
OUT_SIZE_Z = 20
OUT_SIZE = [40, 136, 136]
IN_CHANNELS = 1
OUT_CHANNELS = 2
BATCH_SIZE = 2
INIT_FT_NUM = 64

# [NETWORK]
USEALLGPU = True
GPU_DEVICE = [0]
NET_TYPE = "MSNET0"
NET_NAME = "MSNET_TC32"
DOWNSAMPLE_TWICE = True
CLASS_NUM = 2

# [TRAINING]
LEARNING_RATE = 0.0001
START_EPOCH = 0
START_ITERA = 0
END_EPOCH = 20000
SNAPSHOT_EPOCH = 5
VAL_EPOCH = 5

DECAY_STEP = 15
DECAY_RATE = 0.9

# FOR LOAD THE TRAINED MODEL
EPOCH_MODEL_SAVE_PREFIX = "./history/RMS/saved_models3/model_epoch_"
ITERA_MODEL_SAVE_PREFIX = "./history/RMS/itera_saved_models3/model_itera_"
VAL_SEG_CSV_PATH = "./history/RMS/history_RMS3.csv"
HARD_VAL_SEG_CSV_PATH = "./history/RMS/hard_history_RMS3.csv"
MODEL_SAVE_PATH = "./history/RMS/saved_models3"
IMAGE_SAVE_PATH = "./history/RMS/result_images3"
ITERA_IMAGE_SAVE_PATH = "./history/RMS/hard_result_images3"


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Yuan Network")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME,
                        help="The name of the dataset.")
    parser.add_argument("--train-image-path", type=str, default=TRAIN_IMAGE_PATH,
                        help="Path to the directory containing the train image.")
    parser.add_argument("--train-label-path", type=str, default=TRAIN_LABEL_PATH,
                        help="Path to the directory containing the train label.")
    parser.add_argument("--train-reg-path", type=str, default=TRAIN_REG_PATH,
                        help="Path to the directory containing the train label.")

    parser.add_argument("--val-image-path", type=str, default=VAL_IMAGE_PATH,
                        help="Path to the directory containing the validation image.")
    parser.add_argument("--val-label-path", type=str, default=VAL_LABEL_PATH,
                        help="Path to the directory containing the validation label.")
    parser.add_argument("--in-size-x", type=int, default=IN_SIZE_X,
                        help="The input size of x axis of the volume.")
    parser.add_argument("--in-size-y", type=int, default=IN_SIZE_Y,
                        help="The input size of y axis of the volume.")
    parser.add_argument("--in-size-z", type=int, default=IN_SIZE_Z,
                        help="The input size of z axis of the volume.")
    parser.add_argument("--in-size", type=int, default=IN_SIZE,
                        help="The input patch size of the volume.")
    parser.add_argument("--out-size", type=int, default=OUT_SIZE,
                        help="The input patch size of the volume.")
    parser.add_argument("--out-size-x", type=int, default=OUT_SIZE_X,
                        help="The output size of x axis of the volume.")
    parser.add_argument("--out-size-y", type=int, default=OUT_SIZE_Y,
                        help="The output size of y axis of the volume.")
    parser.add_argument("--out-size-z", type=int, default=OUT_SIZE_Z,
                        help="The output size of z axis of the volume.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--in-channels", type=int, default=IN_CHANNELS,
                        help="")
    parser.add_argument("--out-channels", type=int, default=OUT_CHANNELS,
                        help="")

    parser.add_argument("--init-ft-num", type=int, default=INIT_FT_NUM,
                        help="")

    parser.add_argument("--useallgpu", type=str, default=USEALLGPU,
                        help=".")
    parser.add_argument("--gpu-device", type=str, default=GPU_DEVICE,
                        help=".")
    parser.add_argument("--net-type", type=str, default=NET_TYPE,
                        help=".")
    parser.add_argument("--net-name", type=str, default=NET_NAME,
                        help=".")
    parser.add_argument("--downsample-twice", type=str, default=DOWNSAMPLE_TWICE,
                        help=".")
    parser.add_argument("--class-num", type=int, default=CLASS_NUM,
                        help="Path to the file listing the images in the target dataset.")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--start-iter", type=int, default=START_ITERA,
                        help="The start iter.")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="The start epoch.")
    parser.add_argument("--end-epoch", type=int, default=END_EPOCH,
                        help="The end epoch.")
    parser.add_argument("--snapshot-epoch", type=int, default=SNAPSHOT_EPOCH,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--val-epoch", type=int, default=SNAPSHOT_EPOCH,
                        help="Validation summaries and checkpoint every often..")
    parser.add_argument("--decay-rate", type=float, default=DECAY_RATE,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--decay-step", type=int, default=DECAY_STEP,
                        help="The step of regularisation parameter for L2-loss.")
    parser.add_argument("--epoch-model-save-prefix", type=str, default=EPOCH_MODEL_SAVE_PREFIX,
                        help="The prefix name of model save by epoch.")
    parser.add_argument("--itera-model-save-prefix", type=str, default=ITERA_MODEL_SAVE_PREFIX,
                        help="The prefix name of model save by iteration.")
    parser.add_argument("--val-seg-csv-path", type=str, default=VAL_SEG_CSV_PATH,
                        help="Where to save the validation csv file.")
    parser.add_argument("--hard-val-seg-csv-path", type=str, default=HARD_VAL_SEG_CSV_PATH,
                        help="Where to save the HARD validation csv file.")
    parser.add_argument("--model-save-path", type=str, default=MODEL_SAVE_PATH,
                        help="Where to save the model.")
    parser.add_argument("--image-save-path", type=str, default=IMAGE_SAVE_PATH,
                        help="Where to save the image.")
    parser.add_argument("--itera_image-save-path", type=str, default=ITERA_IMAGE_SAVE_PATH,
                        help="Where to save the image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(args.class_num)
