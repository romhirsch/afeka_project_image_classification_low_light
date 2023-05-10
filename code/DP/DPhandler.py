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
import cv2

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
    enhanced_image_out = np.zeros(data_lowlight.shape)
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


def show_samples(array_of_images, label):
    n = array_of_images.shape[0]
    total_rows = 1+int((n-1)/5)
    total_columns = 5
    fig = plt.figure()
    gridspec_array = fig.add_gridspec(total_rows, total_columns)

    for i, (img, label) in enumerate(zip(array_of_images, label)):
        row = int(i/5)
        col = i % 5
        ax = fig.add_subplot(gridspec_array[row, col])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(classes[int(label)])
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

def preprocess(p=0.5):
    def _preprocess(x):

        #tf.image.random_contrast(x, 0.5, 1)
        if tf.random.uniform([1], 0, 1, seed=seed_aug) < p:
            alpha = float(tf.random.uniform([1], 0.8, 1, seed=seed_aug)[0])
            beta = float(tf.random.uniform([1], 0.5, 1, seed=seed_aug)[0])
            gamma = float(tf.random.uniform([1], 2, 7, seed=seed_aug)[0])
            var_noise = float(tf.random.uniform([1], 0.00001, 0.001, seed=seed_aug)[0])
            img_dark = low_light_transform(np.float32(x) / 255, alpha, beta, gamma)
            #x = img_dark
            img_dark = blur(img_dark)
            x = read_noise(img_dark, var_noise)
            x = np.clip(x*255, 0, 255).astype(np.uint8)
            # augmenter = Augmenter(local_mask=(120, 160), global_mask=(40, 80))
            # x = np.squeeze(augmenter.augment_image(x))

            return preprocess_input(x)
        return preprocess_input(x)
    return _preprocess
#preprocess = preprocess(p=1)

class Environment(Enum):
    TRANSFER = 1
    FINETURNING = 2

class DPhandler(object):

    def __init__(self, classes, num_epochs, early_stop_patience=10, batch_training=100, batch_valid=100,
                 image_size=224, channels=3, trainable_all=False, lr=0.001, name='dp',
                 checkpoint_path=r"E:\dataset\checkpoints", figure_path='', probability_dark=0, pdropout=0.3):
        """
             Initialize the DPhandler class with various parameters for training a deep learning model.

             Parameters:
             - classes: list of class labels
             - num_epochs: number of epochs to run during training
             - early_stop_patience: number of epochs to wait before early stopping if validation loss does not improve
             - batch_training: batch size for training data
             - batch_valid: batch size for validation data
             - image_size: size of the images (assumed to be square)
             - channels: number of channels in the images (e.g. 3 for RGB images)
             - trainable_all: whether to set all layers as trainable or only the top classification layers
             - lr: learning rate for the optimizer
             - name: name of the model
             - checkpoint_path: directory to save checkpoints to
             - figure_path: directory to save figures to
             - image_resize: size to resize images to before feeding them into the model
        """
        self.name = name
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.classes = classes
        self.num_classes = len(classes)
        self._channels = channels
        self.image_size = image_size
        self.batch_training = batch_training
        self.batch_valid = batch_valid
        self.checkpoint_path = os.path.join(checkpoint_path, self.name)
        self.trainable_all = trainable_all
        self.lr = lr
        self.create_relavant_folders(figure_path)
        self.probability_dark = probability_dark
        self.pdropout = pdropout

    def create_relavant_folders(self, figure_path):
        """
        Create the directories for storing checkpoints and figures if they do not already exist.

        Parameters:
        - figure_path: directory to save figures to
        """
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.figure_path = figure_path
        if self.figure_path:
            if not os.path.exists(self.figure_path):
                os.mkdir(self.figure_path)


    def pretrain(self, model, weights, include_top=False):
        """
        Load a pre-trained model.

        Parameters:
        - model: pre-trained model to use
        - weights: weights for the pre-trained model
        - include_top: whether to include the top classification layers in the pre-trained model
        """
        if include_top:
            self.pretrain_model = model(include_top=include_top, weights=weights)
            self.model = self.pretrain_model
        else:
             self.pretrain_model = model(include_top=include_top, pooling='avg', weights=weights)


    def create_model(self, classification_layers):
        # Initialize the model
        self.model = Sequential()
        # Add the pretrained model to the model
        self.model.add(self.pretrain_model)
        self.model.add(tf.keras.layers.Dropout(self.pdropout))
        # Add the classification layers to the model
        for layer in classification_layers:
            self.model.add(layer)
        # Set all layers to be trainable if trainable_all is True,
        # otherwise set the first layer to be non-trainable
        if self.trainable_all:
            for layer in self.model.layers:
                layer.trainable = True
        else:
            self.model.layers[0].trainable = False
        # Print a summary of the model
        self.model.summary()
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=self.lr),
                           loss='categorical_crossentropy', metrics=['accuracy',
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall(),
                                                                     tfa.metrics.F1Score(num_classes=len(classes),
                                                                                         average='macro')
                                                                     ])
        # tf.keras.metrics.TrueNegatives(),
        # tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(),
        # tf.keras.metrics.FalseNegatives()
        # Print which layers are trainable
        print('layers trainable:')
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name, layer.trainable)

    @staticmethod
    def Create_df_dataset_from_directories(directories, percent=1):
        # If percent is not an iterable, set it to
        # a list of the same value repeated for each directory
        if not isinstance(percent, Iterable):
            percent = [percent]*len(directories)
        df = pd.DataFrame({"Images": [], "Labels": []})
        # Loop through the directories
        for i, directory in enumerate(directories):
            images = []
            labels = []
            if os.path.exists(directory):
                # Get a list of subdirectories in the directory
                sub_dirs = os.listdir(directory)
                # Keep only the subdirectories that are in the classes list
                sub_dirs = [v for v in sub_dirs if v in classes]
                # Loop through the subdirectories
                for sub_dir in sub_dirs:
                    # Get the absolute path to the subdirectory
                    abs_path = os.path.join(directory, sub_dir)
                    # Get a list of image names in the subdirectory
                    image_list = os.listdir(abs_path)
                    # If percent is not equal to 1,
                    # select a random subset of images
                    if percent[i] != 1:
                        image_number_choose = int(len(image_list)*percent[i])
                        image_list = random.choices(image_list, k=image_number_choose)
                    image_list = list(map(lambda x: os.path.join(abs_path, x), image_list))
                    images.extend(image_list)
                    labels.extend([sub_dir] * len(image_list))
            # Create a temporary dataframe for
            # the images and labels in the current directory
            df_temp = pd.DataFrame({"Images": images, "Labels": labels})
            df = pd.concat([df, df_temp])
        return df

    @staticmethod
    def loadDataset_df_ds(df, preprocess, classes, BATCH_SIZE=1, image_size=224):
        #self.df_dataset = shuffle(df, random_state=0)
        data_generator = ImageDataGenerator(preprocessing_function=preprocess)

        # flow_from_dataframe generates batches of augmented data
        # Both train & valid folders must have NUM_CLASSES sub-folders
        ds = data_generator.flow_from_dataframe(
            dataframe=df,
            directory=None,
            x_col="Images",
            y_col="Labels",
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE,
            classes=classes,
            shuffle=False,
            seed=123
        )
        return ds

    def loadDataset_df_preprocessing(self, df, preprocess2, dir_val=False):
        self.df_dataset = shuffle(df, random_state=0)
        if self.probability_dark == 0:
            data_generator = ImageDataGenerator(preprocessing_function=preprocess2,
                                                validation_split=0.2,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                brightness_range=(0.8, 1.1),
                                                )
        else:
            data_generator = ImageDataGenerator(preprocessing_function=preprocess2,
                                                validation_split=0.2,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                #brightness_range=[0.2,1.0]
                                                )
        # flow_from_dataframe generates batches of augmented data
        # Both train & valid folders must have NUM_CLASSES sub-folders

        self.train_generator = data_generator.flow_from_dataframe(
            dataframe=self.df_dataset,
            directory=None,
            x_col="Images",
            y_col="Labels",
            subset="training",
            classes=self.classes,
            batch_size=self.batch_training,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(self.image_size, self.image_size))
        if dir_val:
            data_generator_val = ImageDataGenerator(preprocessing_function=preprocess_input)
            self.validation_generator = data_generator_val.flow_from_directory(
                dir_val,
                classes=self.classes,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_valid,
                seed=42,
                shuffle=True,
                class_mode='categorical')

        else:
            if self.probability_dark != 0:
                val_preprocess = preprocess(p=0.5)
            else:
                val_preprocess = preprocess2
            #data_generator_val = ImageDataGenerator(preprocessing_function=preprocess_input)
            data_generator_val = ImageDataGenerator(preprocessing_function=val_preprocess,#preprocess_input,
                                                validation_split=0.2,
                                                horizontal_flip=False,
                                                vertical_flip=False,
                                                # brightness_range=[0.2,1.0]
                                                )
            self.validation_generator = data_generator_val.flow_from_dataframe(
                dataframe=self.df_dataset,
                directory=None,
                x_col="Images",
                y_col="Labels",
                subset="validation",
                classes=self.classes,
                batch_size=self.batch_valid,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=(self.image_size, self.image_size))
        self.step_per_epoch_train = len(self.train_generator)
        self.step_per_epoch_valid = len(self.validation_generator)
        print(self.batch_training, len(self.train_generator), self.batch_valid, len(self.validation_generator))



    def loadDataset_preprocessing(self, dataset_dir, preprocess):

        data_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                            validation_split=0.25,
                                            horizontal_flip=True,
                                            vertical_flip=True
                                            )
        # flow_From_directory generates batches of augmented data
        # Both train & valid folders must have NUM_CLASSES sub-folders
        self.train_generator = data_generator.flow_from_directory(
            dataset_dir,
            classes=self.classes,
            target_size=(self.image_size, self.image_size),
            batch_size= self.batch_training,
            subset='training',
            shuffle=True,
            seed=42,
            class_mode='categorical')

        self.validation_generator = data_generator.flow_from_directory(
            dataset_dir,
            classes=self.classes,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_valid,
            subset='validation',
            seed=42,
            shuffle=True,
            class_mode='categorical')
        self.step_per_epoch_train = len(self.train_generator)
        self.step_per_epoch_valid = len(self.validation_generator)
        print(self.batch_training, len(self.train_generator), self.batch_valid, len(self.validation_generator))


    def fit(self, lr_sch=False):
        cb_early_stopper = EarlyStopping(monitor='val_loss', patience=self.early_stop_patience)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.checkpoint_path, f'{self.name}.best.h5'), #ModelCheckpoint
                                                         save_weights_only=True,
                                                         monitor='val_loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         verbose=1)
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        if lr_sch:
            lr_sched = LearningRateScheduler(lambda epoch: self.lr * (0.75 ** np.floor(epoch / 2)))
            call_backs = [cp_callback, cb_early_stopper, lr_sched, CustomCallback(PathDatasets.EXDARK_VAL.value, self.batch_valid, self.df_dataset, self.classes)]
        else:
            call_backs = [cp_callback, cb_early_stopper, tensorboard_callback, CustomCallback(PathDatasets.EXDARK_VAL.value, self.batch_valid, self.df_dataset, self.classes)]

        self.fit_history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.step_per_epoch_train,
            epochs=self.num_epochs,
            validation_data=self.validation_generator,
            callbacks=call_backs,
            validation_steps=self.step_per_epoch_valid
            )
        # load best weights
        self.model.load_weights(os.path.join(self.checkpoint_path, f'{self.name}.best.h5'))


    def plot_acc_loss(self):
        plt.figure(1, figsize=(15, 8))
        plt.subplot(211)
        plt.plot(self.fit_history.history['accuracy'])
        plt.plot(self.fit_history.history['val_accuracy'])
        plt.title(f'{self.name} model accuracy and loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'])
        plt.subplot(212)
        plt.plot(self.fit_history.history['loss'])
        plt.plot(self.fit_history.history['val_loss'])
        #plt.title(f'{self.name} model loss')
        plt.ylabel(f'loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, f'{self.name}_acc_loss.png'))
        ind = np.argmax(self.fit_history.history['val_accuracy'])
        return self.fit_history.history['val_accuracy'][ind], self.fit_history.history['dark_val_accuracy'][ind], \
               self.fit_history.history['val_loss'][ind], self.fit_history.history['dark_val_loss'][ind], ind #, self.fit_history.history['aug_val_accuracy'][ind], self.fit_history.history['aug_val_loss'][ind]

    @staticmethod
    def load_dataset(dataset_dir, preprocess, classes, BATCH_SIZE=1, image_size=224, shuffle=False):
        data_generator = ImageDataGenerator(preprocessing_function=preprocess)
        ds = data_generator.flow_from_directory(
            directory=dataset_dir,
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE,
            classes=classes,
            shuffle=shuffle,
            seed=123
        )
        return ds


    def predict(self, ds):
        pred = self.model.predict(ds, steps=len(ds), verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        return pred, predicted_class_indices


    def confuction_mat(self, y_true, y_pred, class_names, ds_name):
        cm = confusion_matrix(y_true, y_pred)
        score = metrics.accuracy_score(y_true, y_pred)
        precision_score = metrics.precision_score(y_true, y_pred, average='macro')
        recall_score = metrics.recall_score(y_true, y_pred, average='macro')
        f1_score = metrics.f1_score(y_true, y_pred, average='macro')
        itemsCount = np.sum(cm, axis=1)
        predict = cm.diagonal()
        accuracy = predict / itemsCount
        class_names_with_acc = [class_name + ' ' + str(round(acc, 3)) for class_name, acc in zip(class_names, accuracy)]
        class_names_found = [class_name for class_name, acc in zip(class_names, accuracy)]

        # Plot confusion matrix
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names_found, fontsize=10)
        ax.xaxis.tick_bottom()
        ax.set_ylabel('True', fontsize=20)
        ax.yaxis.set_ticklabels(class_names_with_acc, fontsize=10)
        plt.yticks(rotation=0)
        plt.title(f'{self.name} {ds_name} Confusion Matrix score={round(score,3)}', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, f'{self.name}_{ds_name}.png'))
        #plt.show(block=False)

        return fig, accuracy, score, precision_score, recall_score, f1_score, class_names_found

    def saved_model(self, path):
        self.model.save(path)


    def evaluate(self, ds, ds_name):
        pred, predicted_class_indices = self.predict(ds)
        #predicted_class_indices[predicted_class_indices==1].shape[0]/predicted_class_indices.shape[0]
        fig, acc, score, precision_score, recall_score, f1_score, class_names_found = self.confuction_mat(ds.labels, predicted_class_indices, list(ds.class_indices.keys()), ds_name)
        return np.round(acc, 3), np.round(score, 3), round(precision_score, 3), round(recall_score, 3), round(f1_score, 3), class_names_found


    def plotLR(self):
        learning_rate = self.fit_history.history['lr']
        epochs = range(1, len(learning_rate) + 1)
        fig = plt.figure()
        plt.plot(epochs, learning_rate)
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        return (fig)


    @staticmethod
    def dshistc(ds):
        fig, ax = plt.subplots(1)
        ax.set_ylabel('ds')
        h1 = ax.hist(np.array(list(ds.class_indices.keys()))[ds.labels], bins=len(ds.class_indices.keys()))
        return h1


    def dshist(self):
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].set_ylabel('train ds')
        ax[1].set_ylabel('valid ds')
        h1 = ax[0].hist(np.array(list(self.train_generator.class_indices.keys()))[self.train_generator.labels])
        h2 = ax[1].hist(np.array(list(self.validation_generator.class_indices.keys()))[self.validation_generator.labels])
        return h1, h2


    def load_model(self, path):
        # Recreate the exact same model, including its weights and the optimizer
        self.model = tf.keras.models.load_model(path)
        # Show the model architecture
        self.model.summary()


def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256])


def load_saved_model(path, ModelName, figure_path):
    if os.path.exists(path):
        dp = DPhandler(name=ModelName,
                       classes=classes,
                       num_epochs=20,
                       early_stop_patience=3,
                       batch_training=32,
                       batch_valid=32,
                       trainable_all=True,
                       figure_path=figure_path)

        dp.load_model(path)
    else:
        print(f'saved model no exist {path}')
        return
    return dp


def update_df_summery(df_summery, dp, dataset_name, acc, acc_per_class):
    if np.any((df_summery['model'] == dp.name) & (df_summery['dataset'] == dataset_name)):
        #df_summery.loc[(df_summery['model'] == dp.name) & (df_summery['dataset'] == dataset_name), ['acc'] + classes] = [acc] + list(acc_per_class)
        df_summery.loc[len(df_summery), :] = [dp.name, dataset_name, 0, acc] + list(acc_per_class)

    else:
        df_summery.loc[len(df_summery), :] = [dp.name, dataset_name, 0, acc] + list(acc_per_class)
    return df_summery


def split_ds(path):
    df_ex = DPhandler.Create_df_dataset_from_directories(directories=[PathDatasets.EXDARK.value])
    path_train = r"E:\dataset\ExDark_train"
    path_test = r"E:\dataset\ExDark_test"
    p = 0.25
    for label in df_ex['Labels'].unique():
        df_temp = df_ex[df_ex['Labels'] == label]
        df_temp = shuffle(df_temp)
        n = df_temp.shape[0]
        ind = int(p*n)
        df_test = df_temp[:ind]
        df_train = df_temp[ind:]

        if not os.path.exists(os.path.join(path_train, label)):
            label_path_train = os.path.join(os.path.join(path_train, label))
            os.mkdir(label_path_train)
        if not os.path.exists(os.path.join(path_test, label)):
            label_path_test = os.path.join(os.path.join(path_test, label))
            os.mkdir(label_path_test)

        for path_img_test in df_test['Images']:
            file_test = os.path.basename(path_img_test)
            shutil.copyfile(path_img_test, os.path.join(label_path_test, file_test))

        for path_img_train in df_train['Images']:
            file_train = os.path.basename(path_img_train)
            shutil.copyfile(path_img_train, os.path.join(label_path_train, file_train))

def load_exdark_datasets():
    # load test Datesets:
    df_dataset = DPhandler.\
        Create_df_dataset_from_directories([PathDatasets.EXDARK.value], percent=1)
    files_exdark = [os.path.basename(f) for f in df_dataset['Images']]
    df_dataset.loc[:, 'Light'] = 0
    df_dataset.loc[:, 'Name'] = files_exdark

    df = pd.read_csv((r"E:\dataset\metadata.txt"), delimiter=' ')
    for i, Name in enumerate(df_dataset['Name'].to_numpy()):
        mask = df['Name'].isin([Name])
        if np.any(mask):
            ind_mask = np.where(mask)[0]
            df_dataset.loc[i, 'Light'] = int(df.loc[ind_mask, 'Light'])
    # load ExDark all types
    # Light column: Low(1), Ambient(2), Object(3), Single(4),
    # Weak(5), Strong(6), Screen(7), Window(8), Shadow(9), Twilight(10)
    low_df = df_dataset[df_dataset['Light'] == 1].copy()
    low_df.reset_index(inplace=True)
    Ambient_df = df_dataset[df_dataset['Light'] == 2].copy()
    Ambient_df.reset_index(inplace=True)
    object_df = df_dataset[df_dataset['Light'] == 3].copy()
    object_df.reset_index(inplace=True)
    Single_df = df_dataset[df_dataset['Light'] == 4].copy()
    Single_df.reset_index(inplace=True)
    weak_df = df_dataset[df_dataset['Light'] == 5].copy()
    weak_df.reset_index(inplace=True)
    Strong_df = df_dataset[df_dataset['Light'] == 6].copy()
    Strong_df.reset_index(inplace=True)
    Screen_df = df_dataset[df_dataset['Light'] == 7].copy()
    Screen_df.reset_index(inplace=True)
    Window_df = df_dataset[df_dataset['Light'] == 8].copy()
    Window_df.reset_index(inplace=True)
    Shadow_df = df_dataset[df_dataset['Light'] == 8].copy()
    Shadow_df.reset_index(inplace=True)
    Twilight_df = df_dataset[df_dataset['Light'] == 10].copy()
    Twilight_df.reset_index(inplace=True)
    ds_ExDark_low = DPhandler.loadDataset_df_ds(low_df, preprocess_input, classes)
    ds_ExDark_ambient = DPhandler.loadDataset_df_ds(Ambient_df, preprocess_input, classes)
    ds_ExDark_object = DPhandler.loadDataset_df_ds(object_df, preprocess_input, classes)
    ds_ExDark_Single = DPhandler.loadDataset_df_ds(Single_df, preprocess_input, classes)
    ds_ExDark_weak = DPhandler.loadDataset_df_ds(weak_df, preprocess_input, classes)
    ds_ExDark_strong = DPhandler.loadDataset_df_ds(Strong_df, preprocess_input, classes)
    ds_ExDark_screen = DPhandler.loadDataset_df_ds(Screen_df, preprocess_input, classes)
    ds_ExDark_window = DPhandler.loadDataset_df_ds(Window_df, preprocess_input, classes)
    ds_ExDark_shadow = DPhandler.loadDataset_df_ds(Shadow_df, preprocess_input, classes)
    ds_ExDark_Twilight = DPhandler.loadDataset_df_ds(Twilight_df, preprocess_input, classes)
    ds_exdark_tests = {'low': ds_ExDark_low, 'ambient': ds_ExDark_ambient, 'object': ds_ExDark_object,
                       'single': ds_ExDark_Single, 'weak': ds_ExDark_weak, 'strong': ds_ExDark_strong,
                       'screen': ds_ExDark_screen,
                       'window': ds_ExDark_window, 'shadow': ds_ExDark_shadow, 'Twilight': ds_ExDark_Twilight}
    return ds_exdark_tests

def load_test_dataset(preprocess, only_dark=False, val=False):
    # load test datasets
    #ds_exdark_tests= load_exdark_datasets()
    #df_test = DPhandler.Create_df_dataset_from_directories([PathDatasets.COCO2017_TEST.value], percent=1)
    ds_aug1 = DPhandler.load_dataset(PathDatasets.coco_aug1.value, preprocess, classes)
    ds_aug2 = DPhandler.load_dataset(PathDatasets.coco_aug2.value, preprocess, classes)
    ds_aug3 = DPhandler.load_dataset(PathDatasets.coco_aug3.value, preprocess, classes)
    ds_aug4 = DPhandler.load_dataset(PathDatasets.coco_aug4.value, preprocess, classes)
    ds_augs = [ds_aug1, ds_aug2, ds_aug3, ds_aug4]
    ds_ExDark = DPhandler.load_dataset(PathDatasets.EXDARK.value, preprocess_input, classes)
    ds_ExDark_test = DPhandler.load_dataset(PathDatasets.EXDARK_TEST.value, preprocess, classes)

    if only_dark:
        ds_tests = [ds_ExDark_test] + ds_augs #+ list(ds_exdark_tests.values())
        ds_names = ['ExDark_test'] + ['level1', 'level2', 'level3', 'level4'] #+ list(ds_exdark_tests.keys())
    else:
        if val:
            ds_tests = [DPhandler.load_dataset(PathDatasets.EXDARK_VAL.value, preprocess, classes)]
            ds_names = ['ExDark_val']
        else:
            ds_test = DPhandler.load_dataset(PathDatasets.COCO2017_TEST.value, preprocess, classes)
            ds_tests = [ds_test, ds_ExDark_test] + ds_augs #+ list(ds_exdark_tests.values())
            ds_names = ['Test', 'ExDark_test'] + ['level1', 'level2', 'level3', 'level4'] #+ list(ds_exdark_tests.keys())

    return ds_tests, ds_names


def evaluate_model_on_datasets(dp, df_summery, ds_tests, ds_names, ModelName):
    """
    Evaluate the model on the given datasets and update the summary dataframe with the results.

    Parameters:
    df_summery: Pandas DataFrame
        DataFrame to store the evaluation results for each dataset and model.
    ds_tests: list of Dataset objects
        List of datasets on which to evaluate the model.
    ds_names: list of strings
        List of names for the datasets in ds_tests.
    ModelName: string
        Name of the model being evaluated.
    """
    for ds_test, ds_name in zip(ds_tests, ds_names):
        # Evaluate the model on the current dataset
        acc_per_class, acc, precision_score, recall_score, f1_score, class_names_found = dp.evaluate(ds_test, ds_name)
        print(ds_name)
        print('Accuracy: ', acc, 'F1-score', f1_score)
        # Find the next empty row in the summary dataframe
        ind = 0 if np.all(df_summery['Model'].isna()) else df_summery[~df_summery['Model'].isna()].index[-1] + 1

        # Update the summary dataframe with the evaluation results
        df_summery.iloc[ind, 0:7] = [ModelName, ds_name, 0, acc, precision_score, recall_score, f1_score]
        df_summery.loc[ind, class_names_found] = list(acc_per_class)
    print(df_summery.iloc[:ind + 1, 1:7])
    plt.close('all')
    return df_summery

def dark_ratio(params, do_train, real_dark_train, ds_tests, ds_names, train_directories, df_summery, ratio):
    ModelNames = [params['ModelName'] + f'_{10 - i}_{i}' for i in (ratio//10)]
    probability_darks = ratio/100#np.arange(.0, 1.2, 0.2)
    percent = 1
    for probability_dark, ModelName in zip(probability_darks, ModelNames):
        #probability_dark = 0.3
        if real_dark_train:
            #ModelName = 'EfficientNetV2B0_FineTurning_real_dark'
            train_directories = [PathDatasets.COCO2017_TRAIN.value, PathDatasets.EXDARK_TRAIN.value]
            percent = [0.56 - 0.56 * probability_dark, probability_dark]
            probability_dark = 0

        print(colored(f"{ModelName}", 'green'))
        checkpoint_path, figure_path,\
        saved_models_path = get_directories_and_paths(ModelName)
        if do_train == False:
            # load our trained model
            dp = load_saved_model(saved_models_path, ModelName, figure_path)
        else:
            dp, df_summery = run_train(params['image_size'], params['weights'], probability_dark, ModelName,
                           train_directories, params['num_epochs'], params['batch_training'],
                           params['batch_valid'], params['learning_rate'], params['pretrain_model'],
                           params['lr_schedule'], params['trainable'], params['early_stop_patience'],
                           params['classificationLayers'], percent, df_summery)
        # evaluate accuracy and confusion matrix
        df_summery = evaluate_model_on_datasets(dp, df_summery, ds_tests, ds_names, ModelName)
    df_summery.to_excel(f"{dp.name}.xlsx")

def run_one(params, do_train, train_directories, ds_tests, ds_names, df_summery, do_test=True, save_model=True):
    print(colored(f"{params['ModelName']}", 'green'))
    checkpoint_path, figure_path, saved_models_path = get_directories_and_paths(params['ModelName'])
    if do_train == False:
        # load our trained model
        dp = load_saved_model(saved_models_path, params['ModelName'], figure_path)
        #tf.keras.utils.plot_model(dp.model, show_shapes=False, show_dtype=False, show_layer_names=True, show_layer_activations=True, to_file='model_plot1.png')
    else:
        dp, df_summery = run_train(params['image_size'], params['weights'], params['probability_dark'], params['ModelName'],
                       train_directories, params['num_epochs'], params['batch_training'],
                       params['batch_valid'], params['learning_rate'], params['pretrain_model'],
                       params['lr_schedule'], params['trainable'], params['early_stop_patience'],
                       params['classificationLayers'], 1, df_summery, save_model)
    # evaluate accuracy and confusion matrix
    if do_test:
        df_summery = evaluate_model_on_datasets(dp, df_summery, ds_tests, ds_names, params['ModelName'])
        df_summery.to_excel(f"{dp.name}.xlsx")
    return df_summery

def run_train(image_size, weights, probability_dark, ModelName,
              directories, num_epochs, batch_training,
              batch_valid, learning_rate, pretrain_model,
              lr_schedule, trainable, early_stop_patience,
              classificationLayers, percent, df_summery, save_model=True, pdropout=0.3):
    # Initialize the preprocessing object
    mypreprocess = preprocess(p=probability_dark)
    # Set the path for the figures
    figure_path = os.path.join(file_path, 'figures')
    figure_path = os.path.join(figure_path, ModelName)
    # Set the path for the saved models
    Path = r"E:\dataset\models_saved"
    saved_models_path = os.path.join(Path, ModelName + ".h")

    # Initialize the DPhandler object
    dp = DPhandler(name=ModelName,
                   image_size=image_size,
                   classes=classes,
                   num_epochs=num_epochs,
                   early_stop_patience=early_stop_patience,
                   batch_training=batch_training,
                   batch_valid=batch_valid,
                   trainable_all=trainable,
                   lr=learning_rate,
                   figure_path=figure_path,
                   probability_dark=probability_dark,
                   pdropout=pdropout)
    # Load the pretrained model
    dp.pretrain(pretrain_model, weights=weights)

    # Create the dataset
    df_dataset = dp.Create_df_dataset_from_directories(directories, percent=percent) #[0.6-0.6*0.1, 0.1]
    # Load the dataset and preprocess the images
    dp.loadDataset_df_preprocessing(df_dataset, mypreprocess)
    # Create a dataframe with the number of images per class
    df = pd.DataFrame({classes[i]:[df_dataset['Labels'][df_dataset['Labels'] == classes[i]].count()] for i in range(len(classes))})
    #dp.loadDataset_preprocessing(directories[0], preprocess)
    # Fit the model
    # Create the model
    #with STRATEGY.scope():
    dp.create_model(classificationLayers)
    dp.fit(lr_schedule)
    # Plot the accuracy and loss curves and get the validation accuracy
    acc_val, acc_val_exdark, loss_val, loss_dark_val, best_epoc = dp.plot_acc_loss() #, acc_aug, loss_aug = dp.plot_acc_loss()
    ind = 0 if np.all(df_summery['Model'].isna()) else df_summery[~df_summery['Model'].isna()].index[-1]+1
    df_summery.loc[ind, 'Dataset'] = 'Train'
    df_summery.loc[ind, 'acc_val'] = np.round(acc_val, 2)
    df_summery.loc[ind, 'acc_val_exdark'] = np.round(acc_val_exdark, 2)
    df_summery.loc[ind, 'loss_val'] = loss_val
    df_summery.loc[ind, 'loss_dark_val'] = loss_dark_val
    # df_summery.loc[ind, 'acc_aug'] = acc_aug
    # df_summery.loc[ind, 'loss_aug'] = loss_aug
    df_summery.loc[ind, 'Model'] = dp.name
    df_summery.loc[ind, 'best_epoc'] = best_epoc
    if save_model:
        dp.saved_model(saved_models_path)
    return dp, df_summery

def get_model_parameters():
    dict_parameters = {
    'ModelName': 'EfficientNetV2B0_GS6',#'EfficientNetV2B0_GS012345678910', #'EfficientNetV2B0_GS',#'EfficientNetV2B0_FineTurning',EfficientNetV2B0_FineTurning_real_dark
    'pretrain_model': EfficientNetV2B0,
    'classificationLayers': [Flatten(), Dense(len(classes), activation='softmax')],
    'weights': 'imagenet',
    'image_size': 224,
    'num_epochs': 30,
    'batch_training': 16,
    'batch_valid': 16,
    # learning rate
    'lr_schedule': False,
    'learning_rate': 1e-4,
    # Adam params
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-07,
    # preprocessing params
    'probability_dark': 0.5,
    # fine-turing (trainable=True)
    'trainable': True,
    # call back function params
    'early_stop_patience': 3,
    'pdropout': 0.3,

    }
    return dict_parameters

def edit_parameters_grid_search(dict_edit : dict, dict_parameters : dict, i):
    for k,v in dict_edit.items():
        if isinstance(v, Iterable):
            dict_parameters[k] = v[i]
        else:
            dict_parameters[k] = v
    return dict_parameters

def get_directories_and_paths(ModelName):
    checkpoint_path = r"E:\dataset\checkpoints"
    Path = r"E:\dataset\models_saved"
    figure_path = os.path.join(file_path, 'figures')
    figure_path = os.path.join(figure_path, ModelName)
    saved_models_path = os.path.join(Path, ModelName + ".h")
    return checkpoint_path, figure_path, saved_models_path

def plot_exmaple_ds(path):
    ds_test = DPhandler.load_dataset(path, preprocess_input, classes, BATCH_SIZE=15, shuffle=True)
    for x_batch, y_batch in ds_test:
        label_ind = np.argmax(y_batch, axis=1)
        show_samples(np.uint8(x_batch), label_ind)
        break

class Test_options(Flag):
    """
    RATIO: running the "dark_ratio" function on the dataset, with a specified ratio
    ONE_TEST: running the "run_one" function on the dataset
    GRID_SEARCH: running the test with different parameters specified in "grid_search_params"
    Enhance: running an enhancement method
    """
    RATIO = auto()
    ONE_TEST = auto()
    GRID_SEARCH = auto()
    Enhance = auto()
    MULTI_MODELS = auto()

def update_tensorboard_hparams(params, df_summery, matric_mode='train'):
    if matric_mode == 'train':
        params_save = params.copy()
        matrics = ['acc_val', 'acc_val_exdark', 'loss_val', 'loss_dark_val'] #'loss_aug', 'acc_aug']
        params_save.pop('classificationLayers')
        params_save.pop('pretrain_model')
        params_save.pop('trainable')
        params_save.pop('batch_training')
        params_save.pop('batch_valid')
        params_save['batch_size'] = bz
        params_save['best_epoc'] = int(df_summery.loc[0, 'best_epoc'])
    elif matric_mode == 'test':
        params_save = {'ModelName': params['ModelName']}
        matrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    dict_metric = dict()
    for dataset_i in df_summery['Dataset'].dropna().unique():
        for matric in matrics:
            dict_metric[dataset_i + '_' + matric] = float(df_summery.loc[df_summery['Dataset'] == dataset_i, matric])
    for k, v in params_save.items():
        try:
            params_save[k] = float(v)
        except:
            pass
    for k, v in dict_metric.items():
        try:
            params_save[k] = float(v)
        except:
            pass
    writer.add_hparams(params_save, dict_metric)

if __name__ == '__main__':
    params = get_model_parameters()
    train_directories = [PathDatasets.COCO2017_TRAIN.value]
    ratio = np.array([0, 20, 40, 60, 80, 100])
    do_train = True
    real_dark_train = False
    seed_aug = 0
    test_option = Test_options.Enhance
    multi_models = ['EfficientNetV2B0_GS6', 'EfficientNetV2B0_GS7', 'EfficientNetV2B0_GS26', 'EfficientNetV2B0_GS15', 'EfficientNetV2B0_GS28', 'EfficientNetV2B0_GS29']
    multi_models = [f'EfficientNetV2B0_GS103']
    enhance_methods = {'ZeroDCE++': ZeroDCEpp, 'Normal': preprocess_input, 'SLC-LLE': preprocessing_SLCLLE, 'MBLLEN': preprocessing_mbllen,
                       'SSR': preprocessing_retinex_ssr, 'MSR': preprocessing_retinex_msr, 'MSRCP': preprocessing_msrcp,
                       'MSRCR': preprocessing_msrcr, 'LIME': preprocessing_lime,
                       'AHE': preproceessing_adaptive_histogram_equalization,
                       'HE': preprocessing_histogram_equalization, }

    # create dataframe for results
    keys_gen = ['Model', 'Dataset', 'acc_val', 'Accuracy', 'Precision', 'Recall', 'F1-score'] + classes
    df_summery = pd.DataFrame(np.empty((300, len(keys_gen)))*np.nan, columns=keys_gen)
    # plot exmaple:
    #plot_exmaple_ds(PathDatasets.EXDARK_TEST.value)

    # load test datasets
    #ds_tests, ds_names = load_test_dataset(preprocess_input)

    if Test_options.RATIO in test_option:
        ds_tests, ds_names = load_test_dataset(preprocess_input)
        dark_ratio(params, do_train, real_dark_train, ds_tests, ds_names, train_directories, df_summery, ratio)
    if Test_options.ONE_TEST in test_option:
        ds_tests, ds_names = load_test_dataset(preprocess_input)
        run_one(params, do_train, train_directories, ds_tests, ds_names, df_summery)
        if not do_train:
            matric_mode = 'test'
            params_save = {'ModelName': params['ModelName']}
        else:
            params_save = params.copy()
            matric_mode = 'train'
            update_tensorboard_hparams(params_save, df_summery, matric_mode=matric_mode)
            matric_mode = 'test'
            params_save = {'ModelName': params['ModelName']}
            update_tensorboard_hparams(params_save, df_summery, matric_mode=matric_mode)

    if Test_options.GRID_SEARCH in test_option:
        do_train = True
        random_search = False
        if random_search:
            lr_list = np.random.uniform(size=3, low=1e-3, high=1e-5)
            bz_list = np.random.uniform(size=3, low=2**4, high=2**6)
            lr_schedule_list = np.array([True, False])
            probability_dark_list = np.array([0]) #0.25, 0.5, 0.75, 1])
            dropout_list = np.array([0.3])
        else:
            lr_list = np.array([1e-3, 1e-4, 1e-5])
            bz_list = np.array([2 ** 4, 2 ** 5, 2**6])
            lr_schedule_list = np.array([True, False])
            probability_dark_list = np.array([0]) #0.25, 0.5, 0.75, 1])
            dropout_list = np.array([0.3])
        lr_s, bz_s, probability_dark_s, lr_schedule_s, dropout_s = np.meshgrid(lr_list, bz_list, probability_dark_list,
                                                                               lr_schedule_list, dropout_list)
        lr_s = lr_s.flatten()
        bz_s = bz_s.flatten()
        dropout_s = dropout_s.flatten()
        probability_dark_s = probability_dark_s.flatten()
        lr_schedule_s = lr_schedule_s.flatten().astype(bool)
        base_name = 'EfficientNetV2B0_GS'
        ds_tests, ds_names = load_test_dataset(preprocess_input, False, False)
        for i, (lr, bz, pdark, lr_sch, dropout) in enumerate(
                zip(lr_s, bz_s, probability_dark_s, lr_schedule_s, dropout_s)):
            df_summery = pd.DataFrame(np.empty((300, len(keys_gen))) * np.nan, columns=keys_gen)
            params['ModelName'] = 'EfficientNetV2B0_GridSearch'
            print(f'=============== GridSearch % {round(i/bz_s.shape[0], 2)} {i}/{bz_s.shape[0]}===============')
            print(f'learning rate {lr} \nbatch size {bz}  \ndark_probability {pdark} \nlr_sh {lr_sch}')
            start = timer()
            params['ModelName'] = base_name + f'_lr_{round(lr, 5)}' + f'_bz_{bz}' + f'_dark_{pdark}' + f'_lr_s_{lr_sch}' + f'_dropout_{dropout}'
            params['learning_rate'] = lr
            params['batch_training'] = bz
            params['batch_valid'] = bz
            params['probability_dark'] = pdark
            params['lr_schedule'] = lr_sch
            params['pdropout'] = dropout

            df_summery = run_one(params, do_train, train_directories, ds_tests, ds_names, df_summery, do_test=False, save_model=True)
            update_tensorboard_hparams(params, df_summery)
            end = timer()
            print('time elapsed: ', timedelta(seconds=end - start))


    if Test_options.Enhance in test_option:
        # do Enhance image
        do_train = False

        for name_pre, preprocess in enhance_methods.items():
            ds_tests, ds_names = load_test_dataset(preprocess, only_dark=False)
            # create a CLAHE object (Arguments are optional).
            checkpoint_path, figure_path, \
            saved_models_path = get_directories_and_paths(params['ModelName'])
            dp = load_saved_model(saved_models_path, params['ModelName'], figure_path + '_' + name_pre)
            df_summery = evaluate_model_on_datasets(dp, df_summery, ds_tests, ds_names, params['ModelName'] + '_' + name_pre)
            df_summery.to_excel(params['ModelName'] + '_' + name_pre + '.xlsx')


    if Test_options.MULTI_MODELS in test_option:
        for model_name in multi_models:
            df_summery = pd.DataFrame(np.empty((300, len(keys_gen))) * np.nan, columns=keys_gen)
            params['ModelName'] = model_name
            ds_tests, ds_names = load_test_dataset(preprocess_input)
            df_summery = run_one(params, do_train, train_directories, ds_tests, ds_names, df_summery)
            params_save = {'ModelName': params['ModelName']}
            update_tensorboard_hparams(params, df_summery, matric_mode='test')
            df_summery.to_excel(params['ModelName'] + '_' + 'MULTI_MODELS' + '.xlsx')
    pass


# tensorboard dev upload --logdir C:\Users\spring\Desktop\LSTM\runs
# tensorboard dev upload --logdir C:\Users\rom21\OneDrive\Desktop\git_project\Final_project_afeka\afeka_project_image_classification_low_light\code\DP\logs_tensorboard\run1501