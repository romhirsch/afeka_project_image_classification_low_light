from skimage.util.noise import random_noise
import operator
from scipy.ndimage.morphology import distance_transform_edt as dt
import skimage.draw as draw
from numpy import random
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.util.noise import random_noise
from common.common import PathDatasets
from common.common_functions import plot_img, mse, uint8, variance_of_laplacian

def add_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def blur(image):
    return cv2.blur(image,(5,5))

def sharp_blur(image):
    return blur(image) if np.random.random()>0.5 else add_sharpness(image)

def saturation(image):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvImg[..., 1] = hsvImg[..., 1] * np.random.random()*2
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

def contrast(img):
    factor = max(0.2, np.random.random()*2.)
    return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def regular_augmenter(img, gt):
    shapes = len(img.shape)
    img = np.squeeze(img)
    gt = np.squeeze(gt)
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        gt = np.fliplr(gt)
    if np.random.random() > 0.5:
        idx_x = [576, 544, 512, 480]
        idx_y = [768, 736, 704, 672]
        img = cropND(img, (idx_x[np.random.randint(4)], idx_y[np.random.randint(4)], 3))
        gt = cropND(gt, img.shape)
    if np.random.random() > 0.5:
        img = random_noise(img)
    img = img[np.newaxis, ...]
    gt = gt[np.newaxis, ..., np.newaxis]
    assert len(img.shape) == shapes
    assert len(gt.shape) == shapes
    assert img.shape[1] % 32 == 0
    assert img.shape[2] % 32 == 0
    return img, gt


def get_mask(img, minmax=(120, 160)):
    mask = np.zeros_like(img[..., 0])
    x, y = mask.shape
    min_dim = min(x, y)
    if np.random.random() > 0.5: # Circle-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 2))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.disk([random_x, random_y], random_r)
    else: # Ellipse-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 1.5))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.ellipse(random_x, random_y, random_r, random_r*np.random.uniform(low=0.3, high=0.8, size=1)[0],
                                shape=(x,y),rotation=np.random.random()*np.pi*2-np.pi)
    mask[rr, cc] = 1
    mask = dt(mask)
    rv = np.random.randint(minmax[0], minmax[1])
    mask = mask / np.max(mask) * rv
    return mask, rv


def illumination_augmenter(img, global_mask=(40, 80), local_mask=(120, 160)):
    img = np.squeeze(img)

    if np.random.random()<0.33: img=saturation(img)
    elif np.random.random()<0.66: img=sharp_blur(img)
    else:
        img=contrast(img)
        local_mask=(80,120)
        global_mask=(30,60)

    # Only local changes
    if any(x > 0 for x in local_mask):
        mask, ch = get_mask(img, local_mask)
        mask = np.stack((mask,) * 3, axis=-1)
        sign = '-'
        if np.random.random() > 0.5:
            sign = '+'
            img = img + mask
        else:
            img = img - mask

        # Local and global changes
        if any(x > 0 for x in global_mask):
            if np.random.random() > 0.5:
                sign += '+'
            else:
                sign += '-'
            if sign == '--' or sign == '++':
                global_max = global_mask[1]
                global_min = global_mask[0]
            else:
                global_max = global_mask[1] + ch
                global_min = global_mask[0] + ch

            if sign[1] == '+':
                img = img + np.ones_like(img) * np.random.randint(global_min, global_max)
            elif sign[1] == '-':
                img = img + np.ones_like(img) * np.random.randint(global_min, global_max) * -1

    # Only global changes
    elif any(x > 0 for x in global_mask):
        global_min, global_max = global_mask
        sign = [-1, 1]
        img = img + np.ones_like(img) * np.random.randint(global_min, global_max) * sign[np.random.randint(0, 2)]
    img[img > 255] = 255
    img[img < 0] = 0
    return img[np.newaxis, ...]

class Augmenter:
    def __init__(self, local_mask=(120, 160), global_mask=(40, 80)):

        #self.augmenting_prob = augmenting_prob
        self.local_mask = local_mask
        self.global_mask = global_mask
        self.augment_illumination = any(x > 0 for x in list(local_mask) + list(global_mask))

    def augment_image(self, img):
        # if self.augment_illumination and np.random.random() < self.augmenting_prob:
        img = illumination_augmenter(img, self.global_mask, self.local_mask)
        # if self.flip_and_noise and np.random.random() < self.augmenting_prob:
        #     img, gt = regular_augmenter(img, gt)
        return img




def GammaCorr(img, gamma):
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return np.float32(res)/255

def low_light_transform(img, alpha, beta, gamma):
    return beta * GammaCorr(alpha * img, gamma)

def read_noise(img, var):
    return random_noise(img, mode='gaussian', var=var)

def blur(img, size=3):
    return cv2.GaussianBlur(img, (size, size), 0)

def get_Y_hist(img):
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, bin_edges = np.histogram(img_out[..., 0], bins=256, range=(0, 255))
    return y

def bgr_to_bayer(img):
    (height, width) = img.shape[:2]
    (B, G, R) = cv2.split(img)
    bayer = np.empty((height, width), np.uint8)
    # strided slicing for this pattern:
    #   G R
    #   B G
    bayer[0::2, 0::2] = G[0::2, 0::2]  # top left
    bayer[0::2, 1::2] = R[0::2, 1::2]  # top right
    bayer[1::2, 0::2] = B[1::2, 0::2]  # bottom left
    bayer[1::2, 1::2] = G[1::2, 1::2]  # bottom right
    return bayer

def noise_bayer(img, var=0.0001):
    bayer = bgr_to_bayer(img)
    bayer_noised = random_noise(bayer, mode='gaussian', var=var)
    img_noisy = cv2.cvtColor(uint8(bayer_noised), cv2.COLOR_BAYER_GRBG2BGR)
    return img_noisy

def pair_compare(img, target_dark, gamma, alpha, beta):
    img = np.float32(img)/255
    img_dark = low_light_transform(img, alpha, beta, gamma)
    img_dark = blur(img_dark)
    #img_dark = read_noise(img_dark, 0.0001)
    #img_dark = random_noise(img_dark, mode='gaussian', var=0.0001)
    img_dark = uint8(img_dark*255)
    img = uint8(img*255)
    plot_img(img, 'normal light image')
    plot_img(img_dark, 'Synthesis low-light image')
    plot_img(target_dark, 'references low-light image')
    print('blur estimate target: ', variance_of_laplacian(target_dark))
    print('blur estimate synthesis: ', variance_of_laplacian(img_dark))
    yorig = get_Y_hist(img)
    ydark = get_Y_hist(img_dark)
    yreff = get_Y_hist(target_dark)
    fig, ax = plt.subplots(1)
    ax.set_title('histogram Y channel in YCbCr')
    ax.plot(ydark, color='r', label='Y synthesis')
    ax.plot(yorig, color='g', label='Y orig')
    ax.plot(yreff, color='b', label='Y references')
    ax.set_xlabel('Pixel values')
    ax.set_ylabel('No. of Pixels')
    ax.legend()


def find_params_from_target(img, target_dark, alpha=0.9):
    gammas = np.linspace(1.2, 5, 15)
    betas = np.linspace(0.5, 1, 15)  #
    img = np.float32(img / 255)
    target_dark = np.float32(target_dark / 255)
    res = []
    res_ind = []
    xv, yv = np.meshgrid(gammas, betas)
    for i in range(len(xv)):
        for j in range(len(yv)):
                gamma = xv[j, i]
                beta = yv[j, i]
                dark_img = beta * ((alpha * img) ** gamma)
                res.append(mse(dark_img, target_dark))
                res_ind.append([gamma, alpha, beta])
    res = np.array(res)
    min = res.argmin()
    gamma = res_ind[min][0]
    alpha = res_ind[min][1]
    beta = res_ind[min][2]
    return gamma, alpha, beta


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, os.listdir(folder)


def create_dark_images(imgs):
    dark_imgs = []
    for img in imgs:
        augmenter = Augmenter(local_mask=(120, 160), global_mask=(40, 80))
        img_dark = np.squeeze(augmenter.augment_image(img))
        dark_imgs.append(img_dark.copy())
    return dark_imgs


def save_imgs(imgs, list_files, path):
    for img, name in zip(imgs, list_files):
        path_img = os.path.join(path, name)
        cv2.imwrite(path_img, img)



def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


if __name__ == '__main__':
    save_folder = r"E:\coco_aug1_1"
    ds_folder = PathDatasets.COCO_TEST.value
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for i, folder_name in enumerate(os.listdir(ds_folder)):
        print(folder_name)
        folder = os.path.join(ds_folder, folder_name)
        imgs, list_files = load_images_from_folder(folder)
        dark_imgs = create_dark_images(imgs)
        cur_save_folder = os.path.join(save_folder, os.path.basename(folder))
        if not os.path.exists(cur_save_folder):
            os.mkdir(cur_save_folder)
        save_imgs(dark_imgs, list_files, cur_save_folder)





