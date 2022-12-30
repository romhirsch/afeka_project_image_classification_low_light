"""
Illumination-Based Data Augmentation for Robust Background Subtraction
Copyright (c) 2019 Dimitrios SAKKOS, Hubert SHUM and Edmond S. L. HO.
Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (see LICENSE for details)
Written by Dimitrios Sakkos
"""

from skimage.util.noise import random_noise
import operator
from scipy.ndimage.morphology import distance_transform_edt as dt
import skimage.draw as draw
import numpy as np
import cv2
from DP.DPhandler import PathDatasets

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


def regular_augmenter(img):
    shapes = len(img.shape)
    img = np.squeeze(img)
    #gt = np.squeeze(gt)
    # if np.random.random() > 0.5:
    #     img = np.fliplr(img)
    #
    #  #   gt = np.fliplr(gt)
    # if np.random.random() > 0.5:
    #     idx_x = [576, 544, 512, 480]
    #     idx_y = [768, 736, 704, 672]
    #     img = cropND(img, (idx_x[np.random.randint(4)], idx_y[np.random.randint(4)], 3))
      #  gt = cropND(gt, img.shape)
    if np.random.random() > 0.5:
        img = random_noise(img)
    img = img[np.newaxis, ...]
    #gt = gt[np.newaxis, ..., np.newaxis]
    #assert len(gt.shape) == shapes
    return img


def get_mask(img, minmax=(120, 160)):
    mask = np.zeros_like(img[..., 0])
    x, y = mask.shape
    min_dim = min(x, y)
    if np.random.random() > 0.5: # Circle-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 2))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.circle_perimeter(random_x, random_y, random_r)
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
    def __init__(self, local_mask=(120, 160), global_mask=(40, 80),
                 flip_and_noise=False, augmenting_prob=0.67):

        self.augmenting_prob = augmenting_prob
        self.local_mask = local_mask
        self.global_mask = global_mask
        self.flip_and_noise = flip_and_noise
        self.augment_illumination = any(x > 0 for x in list(local_mask) + list(global_mask))

    def augment_image(self, img):
        if self.augment_illumination and np.random.random() < self.augmenting_prob:
            img = illumination_augmenter(img, self.global_mask, self.local_mask)
        if self.flip_and_noise and np.random.random() < self.augmenting_prob:
            img = regular_augmenter(img)
        return img
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, os.listdir(folder)

def save_imgs(imgs, list_files, path):
    for img, name in zip(imgs, list_files):
        path_img = os.path.join(path, name)
        cv2.imwrite(path_img, img)
import matplotlib.pyplot as plt

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256])

def create_dark_images(imgs):
    dark_imgs = []
    for img in imgs:
        augment_local = (120, 160)
        augment_global = (40, 80)

        augmenter = Augmenter(local_mask=augment_local, global_mask=augment_global,
                              flip_and_noise=True)
        dark_imgs.append(augmenter.augment_image(img))
    return dark_imgs

save_folder = r"E:\dataset\aug_"
#img = cv2.imread(r"E:\dataset\lol\eval15\high\778.png")
#ag = Augmenter(brightness=-110, contrast=0.1, noise=False, blur=False, saturation=False)
#x1 = ag.illumination_augmenter(img)
ds_folder = PathDatasets.COCO_TEST.value
for i, folder_name in enumerate(os.listdir(ds_folder)):
    print(folder_name)
    folder = os.path.join(ds_folder, folder_name)
    imgs, list_files = load_images_from_folder(folder)
    dark_imgs = create_dark_images(imgs)
    cur_save_folder = os.path.join(save_folder, os.path.basename(folder))
    if not os.path.exists(cur_save_folder):
        os.mkdir(cur_save_folder)
    save_imgs(dark_imgs, list_files, cur_save_folder)

