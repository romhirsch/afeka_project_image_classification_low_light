import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from MSRCR.src import config
from MSRCR.src.utils import read_show, plot_hist
from MSRCR.src.retinex import MSRCR
from retinex.msrcp import msrcp
from LIME.LIME_CLI import LIME
import random
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from sklearn import metrics
from common.common import *
import os
from collections.abc import Iterable
from tensorflow.python.client import device_lib
from sklearn.utils import shuffle
import shutil
from augmentation.Dark_Augmenter import low_light_transform, blur, read_noise
from colorama import init
from termcolor import colored
# keras applications:
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4, EfficientNetV2B0, ResNet50
from illumination_augmentation.augmenter import Augmenter
from enum import Flag, auto
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import sys
sys.path.append(r'C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\MBBLLEN')
#from ZeroDCEpp.lowlight_test import lowlight


from glob import glob
import numpy as np
import scipy
import keras
import os
import Network
import utls
import time
import cv2
import argparse
import torch
import torch.optim
import os
import sys
sys.path.append(r'C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\SLCLLE')
import lowlight_model
sys.path.append(r'C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\ZeroDCEpp')
import model_zeroDCEpp
def ZeroDCEpp(data_lowlight):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = 1
    # data_lowlight = Image.open(image_path)
    #
    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model_zeroDCEpp.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load(r"C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\ZeroDCEpp\snapshots_Zero_DCE++\Epoch99.pth"))
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)

    end_time = (time.time() - start)

    #print(end_time)
    #image_path = image_path.replace('test_data', 'result_Zero_DCE++')

    # result_path = image_path
    # if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
    #     os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    # import pdb;pdb.set_trace()
    #torchvision.utils.save_image(enhanced_image, result_path)
    enhanced_image_out = np.zeros(img.shape)
    x = np.squeeze(enhanced_image.cpu().detach().numpy())
    enhanced_image_out[:, :, 0] = x[0]
    enhanced_image_out[:, :, 1] = x[1]
    enhanced_image_out[:, :, 2] = x[2]
    enhanced_image_out = np.round(enhanced_image_out * 255)
    return enhanced_image_out

class scl_lle:
  def __init__(self):
    self.SCL_LLE_net = lowlight_model.enhance_net_nopool().cuda()
    self.SCL_LLE_net.load_state_dict(torch.load(r'C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\SLCLLE\checkpoints\SCL-LLE.pth'))

  def test(self, img):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    data_lowlight = img
    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    _, enhanced_image, _ = self.SCL_LLE_net(data_lowlight)

    return enhanced_image
class mbllen:
  def __init__(self):
    self.lowpercent = 5
    self.highpercent = 95
    self.maxrange = 8/10
    self.hsvgamma = 8/10

    self.model_name = 'Syn_img_lowlight_withnoise'
    self.model = Network.build_mbllen((None, None, 3))
    self.model.load_weights(r"C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\MBBLLEN\models\Syn_img_lowlight_withnoise.h5")
    opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    self.model.compile(loss='mse', optimizer=opt)

  def test(self, img):
    img_A = img[np.newaxis, :]

    out_pred = self.model.predict(img_A)

    fake_B = out_pred[0, :, :, :3]
    fake_B_o = fake_B

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
    percent_max = sum(sum(gray_fake_B >= self.maxrange))/sum(sum(gray_fake_B <= 1.0))
    max_value = np.percentile(gray_fake_B[:], self.highpercent)
    if percent_max < (100-self.highpercent)/100.:
        scale = self.maxrange / max_value
        fake_B = fake_B * scale
        fake_B = np.minimum(fake_B, 1.0)

    gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
    sub_value = np.percentile(gray_fake_B[:], self.lowpercent)
    fake_B = (fake_B - sub_value)*(1./(1-sub_value))

    imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, self.hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    fake_B = np.minimum(fake_B, 1.0)
    fake_B = np.minimum(fake_B, 1.0)
    fake_B = np.maximum(fake_B, 0.0)
    #fake_B = np.uint8(fake_B * 255)
    return fake_B

mymbllen = mbllen()
#%%
import tensorboard
writer = SummaryWriter(r'.\logs_tensorboard\gs1303')

init()
file_path = os.path.realpath(__file__)
file_path = file_path.replace(os.path.basename(file_path), '')

np.random.seed(1)


classes = ['airliner',
 'bicycle',
 'Border collie',
 'cup',
 'dining table',
 'elephant',
 'folding chair',
 'golden retriever',
 'military plane',
 'minibus',
 'Persian cat',
 'school_bus',
 'speedboat',
 'tabby cat',
 'yawl']

classes = ["bicycle", "boat", "bus", "car",
           "cat", "dog", "motorcycle", "person"]

def print_array_info(v):
    print("{} is of type {} with shape {} and dtype {}".format(v,
                                                           eval("type({})".format(v)),
                                                           eval("{}.shape".format(v)),
                                                           eval("{}.dtype".format(v))
                                                           ))

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, path_ds, batch_size, df_dataset, classes):
        self.ds = DPhandler.load_dataset(path_ds,
                                         preprocess_input, classes, BATCH_SIZE=batch_size)
        data_generator_val = ImageDataGenerator(preprocessing_function=preprocess(p=0.5),
                                                validation_split=0.2,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                # brightness_range=[0.2,1.0]
                                                )
        self.ds_val_p = data_generator_val.flow_from_dataframe(
            dataframe=df_dataset,
            directory=None,
            x_col="Images",
            y_col="Labels",
            subset="validation",
            classes=classes,
            batch_size=batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(224, 224))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        res = self.model.evaluate(self.ds, verbose=0)
        # logs['dark_val_loss'] = 0#res[0]
        # logs['dark_val_accuracy'] = 0#res[1]
        # logs['dark_val_precision'] = 0#res[2]
        # logs['dark_val_recall'] = 0#res[3]
        logs['dark_val_loss'] = res[0]
        logs['dark_val_accuracy'] = res[1]
        logs['dark_val_precision'] = res[2]
        logs['dark_val_recall'] = res[3]
        # res_val = self.model.evaluate(self.ds, verbose=0)
        # logs['aug_val_loss'] = res_val[0]
        # logs['aug_val_accuracy'] = res_val[1]
        # logs['aug_val_precision'] = res_val[2]
        # logs['aug_val_recall'] = res_val[3]

        # print('exdark loss:', round(logs['dark_val_loss'],4),
        #       ' exdark accuracy:',  round(logs['dark_val_accuracy'],4))

        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))

def show_samples_all(array_of_images, label):
    n = array_of_images.shape[0]
    total_rows = 1+int((n-1)/11)
    total_columns = 11
    fig = plt.figure()
    #gridspec_array = fig.add_gridspec(total_columns, total_rows)
    gridspec_array = fig.add_gridspec(total_rows, total_columns)

    for i, (img, label) in enumerate(zip(array_of_images, label)):
        row = int(i/11)
        col = i % 11
#        ax = fig.add_subplot(gridspec_array[col, row])
        ax = fig.add_subplot(gridspec_array[row, col])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i < total_columns:
             ax.set_title(label)
        ax.imshow(img)

    plt.show()

def show_samples(array_of_images, label):
    n = array_of_images.shape[0]
    total_rows = 1+int((n-1)/10)
    total_columns = 10
    fig = plt.figure()
    gridspec_array = fig.add_gridspec(total_rows, total_columns)

    for i, (img, label) in enumerate(zip(array_of_images, label)):
        row = int(i/10)
        col = i % 10
        ax = fig.add_subplot(gridspec_array[row, col])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(label)
        ax.imshow(img)

    plt.show()
# classes = ['bicycle',
#  'boat',
#  'bottle',
#  'bus',
#  'car',
#  'cat',
#  'chair',
#  'cup',
#  'dining table',
#  'dog',
#  'motorcycle',
#  'person']
#gen_G_low = tf.keras.models.load_model(r"C:\Users\rom21\OneDrive\Desktop\Final_project_afeka\code\augmentation\cyclegan_low.090")
# gen_G_weak = tf.keras.models.load_model(r"C:\Users\rom21\OneDrive\Desktop\Final_project_afeka\code\augmentation\cyclegan_weak.090")
# gen_G_Ambient = tf.keras.models.load_model(r"C:\Users\rom21\OneDrive\Desktop\Final_project_afeka\code\augmentation\cyclegan_Ambient.090")
# gen_G_single = tf.keras.models.load_model(r"C:\Users\rom21\OneDrive\Desktop\Final_project_afeka\code\augmentation\cyclegan_Single.090")

# prediction = gen_G(img, training=False)[0].numpy()
# prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
print(tf.__version__)
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def uint8c(x):
    """custom uint8 - (if val < 0 then val = 0)"""
    x2 = np.copy(x)
    #x2[x2 < 0] = 0
    #x2[x2 > 255] = 255
    x2  = x2.clip(0, 255)
    return np.uint8(x2)

seed_aug = 1

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

def preprocessing_SLCLLE(img):
    mynet = scl_lle()
    #img = cv2.cvtColor(cv2.imread(r"E:\coco_aug_4\bus\004304.jpg"), cv2.COLOR_BGR2RGB)
    x = mynet.test(img)
    x = x.cpu().detach().numpy()
    img2 = np.squeeze(x)
    img_scl = np.zeros(img.shape)
    img_scl[:, :, 0] = img2[0, :, :]
    img_scl[:, :, 1]  = img2[1, :, :]
    img_scl[:, :, 2]  = img2[2, :, :]
    return np.round(img_scl*255)

def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def preprocessing_msrcr(img):
    msrcr_img = MSRCR(img.astype(np.uint8), config.SIGMA_LIST, config.ALPHA, config.BETA, config.G, config.OFFSET)
    return msrcr_img * 1.

def preprocessing_msrcr(img):
    msrcr_img = MSRCR(img.astype(np.uint8), config.SIGMA_LIST, config.ALPHA, config.BETA, config.G, config.OFFSET)
    return msrcr_img * 1.


def preprocessing_lime(img):
    lime = LIME(**{'filePath': 'E:\\coco_aug_2\\bicycle\\004544.jpg',
             'map': False,
             'output': './',
             'iterations': 10,
             'alpha': 2,
             'rho': 2,
             'gamma': 0.7,
             'strategy': 2})
    lime.load(img)
    lime.enhance()
    return lime.R * 1.

def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex
Countimages = 0

def preprocessing_msrcp(img):
    img_msrcp = msrcp(img.astype(np.uint8)) * 1.
    return img_msrcp



def preprocessing_mbllen(img):
    #img = cv2.cvtColor(img/255.0, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    #global Countimages
    img_mbllen = mymbllen.test(img)
    #Countimages+=1

    #utls.imwrite(r'E:/sasa/' + str(Countimages) + '.jpg', cv2.cvtColor(img_mbllen, cv2.COLOR_RGB2BGR))
    #img_mbllen = np.round(cv2.cvtColor(img_mbllen, cv2.COLOR_RGB2BGR)*255, 0)
    img_mbllen = np.round(img_mbllen *255.)

    #img_mbllen = scipy.misc.toimage(img_mbllen * 255, high=255, low=0, cmin=0, cmax=255)

    return img_mbllen

def preprocessing_retinex_ssr(img, variance=300):
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        try:
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break
        except:
            pass
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    #img_retinex = cv2.cvtColor(img_retinex, cv2.COLOR_RGB2BGR)
    return img_retinex

def preprocessing_retinex_msr(img, variance_list=[15, 80, 30]):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        try:

            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break
        except:
            pass
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex

def preproceessing_adaptive_histogram_equalization(img):
    # configure CLAHE
    img = np.uint8(img)
    clahe_model = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(img[:, :, 0])
    colorimage_g = clahe_model.apply(img[:, :, 1])
    colorimage_r = clahe_model.apply(img[:, :, 2])
    img_ahe = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    return img_ahe

def normal(img):
    return img
if __name__ == '__main__':

    images_name = r"person\000000000872.jpg"
    orig_img = cv2.cvtColor(cv2.imread(rf"E:\COCO2017_Test\{images_name}"), cv2.COLOR_BGR2RGB)
    labels = []
    img = cv2.cvtColor(cv2.imread(rf"E:\coco_aug_1\{images_name}"), cv2.COLOR_BGR2RGB) * 1.
    images = np.zeros((11 * 4, img.shape[0], img.shape[1], img.shape[2]))
    psnr_images = np.zeros((11 * 4))
    ssim_images = np.zeros((11 * 4))

    count=0
    for i in range(1, 5):
        img = cv2.cvtColor(cv2.imread(rf"E:\coco_aug_{i}\{images_name}"), cv2.COLOR_BGR2RGB) * 1.
        enhance_methods = {'Input': normal, 'ZeroDCE++': ZeroDCEpp,  'SLC-LLE': preprocessing_SLCLLE, 'MBLLEN': preprocessing_mbllen, 'SSR': preprocessing_retinex_ssr, 'MSR':preprocessing_retinex_msr, 'MSRCP': preprocessing_msrcp,  'MSRCR': preprocessing_msrcr,  'LIME': preprocessing_lime,  'AHE': preproceessing_adaptive_histogram_equalization,  'HE': preprocessing_histogram_equalization, }
        labels = labels + list(enhance_methods.keys())
        for j, (name, func) in enumerate(enhance_methods.items()):
            images[count] = (func(img))
            psnr_images[count] = tf.image.psnr(orig_img, images[count].astype(np.uint8), max_val=255, name=None).numpy()
            ssim_images[count] = tf.image.ssim(tf.expand_dims(orig_img, axis=0), tf.expand_dims(images[count].astype(np.uint8), axis=0) , max_val=255).numpy()

            count+=1

    show_samples_all(images.astype(np.uint8), labels)
    import pandas as pd
    dfssim = pd.DataFrame({"level1": ssim_images[0:11], "level2": ssim_images[11:22], "level3": ssim_images[22:33], "level4": ssim_images[33:44]})
    df = pd.DataFrame({"level1": psnr_images[0:11], "level2": psnr_images[11:22], "level3": psnr_images[22:33], "level4": psnr_images[33:44]})
    x = 5
    pass