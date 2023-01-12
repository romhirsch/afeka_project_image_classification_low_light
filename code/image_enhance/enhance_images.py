import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_two_exmaples(img1, img2):
    fig, ax = plt.subplots(1, 2);
    ax[0].imshow(img1);
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].imshow(img2)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

def preprocessing_histogram_equalization(img):
    img = np.uint8(img)
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

    return equalized_img

def preproceessing_adaptive_histogram_equalization(img):
    # configure CLAHE
    img = np.uint8(img)
    clahe_model = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(img[:, :, 0])
    colorimage_g = clahe_model.apply(img[:, :, 1])
    colorimage_r = clahe_model.apply(img[:, :, 2])
    img_ahe = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    return img_ahe
path = r"E:\coco_aug_4\dog\000000486479.jpg"
img = cv2.imread(path)
img_e = preproceessing_adaptive_histogram_equalization(img)
show_two_exmaples(img, img_e)