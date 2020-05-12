import cv2
import skimage.io as io
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

GAMMA = 0.5
def preprocess_gamma_hist(imgs, gamma=10):
    invGamma = 1.0/gamma
    #build the gamma lookup table for color correctness (grayscale)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    #apply gamma correction that controls the overall brightness
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    plt.figure(1)
    plt.imshow(new_imgs, cmap='gray')
    #apply the histogram equalization to improve the contrast
    new_img = cv2.equalizeHist(new_imgs)
    plt.figure(2)
    plt.imshow(new_img, cmap='gray')
    plt.show()
    print(np.max(new_img))
    return new_img


img = io.imread('oriCvLab/trainCvlab/img/train004.png',as_gray = True)
print(type(img))
preprocess_gamma_hist(img)