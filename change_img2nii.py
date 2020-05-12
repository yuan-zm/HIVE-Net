import nibabel, glob
import numpy as np
from util.tools_self import save_array_as_nii_volume, load_nifty_volume_as_array
from PIL import Image
from metrics import *
from matplotlib import pyplot as plt
import scipy.misc


pdsz = 30
save_filename = 'train_yz_proximity.nii.gz'
img_path = 'oriCvLab/train_lab_c2_yz'

img_arr = glob.glob(str(img_path) + "/*")
img_arr.sort()

img = np.zeros((165,768//2,1024//2)).astype('uint8')
#  lab = np.zeros((165,1024//2,768//2)).astype('uint8')

for i in range(len(img_arr)):
    img_as_img = Image.open(img_arr[i])
    # img_as_img.show()
    img_as_np = np.asarray(img_as_img).astype('uint8')
    # img_as_np = scipy.misc.imresize(img_as_np, 0.5)
    img[:,:,i] = img_as_np

img = np.pad(img, pdsz, mode='symmetric')

save_array_as_nii_volume(np.transpose(img, [2,0,1]), save_filename)

