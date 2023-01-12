
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from tensorflow.keras.models import Sequential
import tensorflow as tf
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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4, EfficientNetV2B0
from illumination_augmentation.augmenter import Augmenter
from enum import Flag, auto

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
                 checkpoint_path=r"E:\dataset\checkpoints", figure_path=''):
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
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
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

    def loadDataset_df_preprocessing(self, df, preprocess, dir_val=False):
        self.df_dataset = shuffle(df, random_state=0)
        data_generator = ImageDataGenerator(preprocessing_function=preprocess,
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
            self.validation_generator = data_generator.flow_from_dataframe(
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
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         save_best_only=True)
        if lr_sch:
            lr_sched = LearningRateScheduler(lambda epoch: self.lr * (0.75 ** np.floor(epoch / 2)))
            call_backs = [cp_callback, cb_early_stopper, lr_sched]
        else:
            call_backs = [cp_callback, cb_early_stopper]

        self.fit_history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.step_per_epoch_train,
            epochs=self.num_epochs,
            validation_data=self.validation_generator,
            callbacks=call_backs,
            validation_steps=self.step_per_epoch_valid)
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
        return np.max(self.fit_history.history['val_accuracy'])

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

        return accuracy, score, precision_score, recall_score, f1_score, class_names_found

    def saved_model(self, path):
        self.model.save(path)


    def evaluate(self, ds, ds_name):
        pred, predicted_class_indices = self.predict(ds)
        #predicted_class_indices[predicted_class_indices==1].shape[0]/predicted_class_indices.shape[0]
        acc, score, precision_score, recall_score, f1_score, class_names_found = self.confuction_mat(ds.labels, predicted_class_indices, list(ds.class_indices.keys()), ds_name)
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
        fig, ax = plt.subplots(2,1, sharex=True)
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

def load_test_dataset(preprocess, only_dark=False):
    # load test datasets
    #ds_exdark_tests= load_exdark_datasets()
    #df_test = DPhandler.Create_df_dataset_from_directories([PathDatasets.COCO2017_TEST.value], percent=1)
    ds_aug1 = DPhandler.load_dataset(PathDatasets.coco_aug1.value, preprocess, classes)
    ds_aug2 = DPhandler.load_dataset(PathDatasets.coco_aug2.value, preprocess, classes)
    ds_aug3 = DPhandler.load_dataset(PathDatasets.coco_aug3.value, preprocess, classes)
    ds_aug4 = DPhandler.load_dataset(PathDatasets.coco_aug4.value, preprocess, classes)
    ds_augs = [ds_aug1, ds_aug2, ds_aug3, ds_aug4]
    #ds_ExDark = DPhandler.load_dataset(PathDatasets.EXDARK.value, preprocess_input, classes)
    ds_ExDark_test = DPhandler.load_dataset(PathDatasets.EXDARK_TEST.value, preprocess, classes)
    if only_dark:
        ds_tests = [ds_ExDark_test] + ds_augs #+ list(ds_exdark_tests.values())
        ds_names = ['ExDark_test'] + ['level1', 'level2', 'level3', 'level4'] #+ list(ds_exdark_tests.keys())
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
        checkpoint_path, figure_path, \
        checkpoint_path, figure_path, saved_models_path = get_directories_and_paths(ModelName)
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

def run_one(params, do_train, train_directories, ds_tests, ds_names, df_summery, do_test=True):
    print(colored(f"{params['ModelName']}", 'green'))
    checkpoint_path, figure_path, \
    checkpoint_path, figure_path, saved_models_path = get_directories_and_paths(params['ModelName'])
    if do_train == False:
        # load our trained model
        dp = load_saved_model(saved_models_path, params['ModelName'], figure_path)
    else:
        dp, df_summery = run_train(params['image_size'], params['weights'], params['probability_dark'], params['ModelName'],
                       train_directories, params['num_epochs'], params['batch_training'],
                       params['batch_valid'], params['learning_rate'], params['pretrain_model'],
                       params['lr_schedule'], params['trainable'], params['early_stop_patience'],
                       params['classificationLayers'], 1, df_summery)
    # evaluate accuracy and confusion matrix
    if do_test:
        df_summery = evaluate_model_on_datasets(dp, df_summery, ds_tests, ds_names, params['ModelName'])
        df_summery.to_excel(f"{dp.name}.xlsx")
    return df_summery

def run_train(image_size, weights, probability_dark, ModelName,
              directories, num_epochs, batch_training,
              batch_valid, learning_rate, pretrain_model,
              lr_schedule, trainable, early_stop_patience,
              classificationLayers, percent, df_summery):
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
                   figure_path=figure_path)
    # Load the pretrained model
    dp.pretrain(pretrain_model, weights=weights)
    # Create the model
    dp.create_model(classificationLayers)
    # Create the dataset
    df_dataset = dp.Create_df_dataset_from_directories(directories, percent=percent) #[0.6-0.6*0.1, 0.1]
    # Load the dataset and preprocess the images
    dp.loadDataset_df_preprocessing(df_dataset, mypreprocess)
    # Create a dataframe with the number of images per class
    df = pd.DataFrame({classes[i]:[df_dataset['Labels'][df_dataset['Labels']==classes[i]].count()] for i in range(len(classes))})
    #dp.loadDataset_preprocessing(directories[0], preprocess)
    # Fit the model
    dp.fit(lr_schedule)
    # Plot the accuracy and loss curves and get the validation accuracy
    acc_val = dp.plot_acc_loss()
    ind = 0 if np.all(df_summery['Model'].isna()) else df_summery[~df_summery['Model'].isna()].index[-1]+1
    df_summery.loc[ind, 'acc_val'] = np.round(acc_val, 2)
    df_summery.loc[ind, 'Model'] = dp.name
    dp.saved_model(saved_models_path)
    return dp, df_summery

def get_model_parameters():
    dict_parameters = {
    'ModelName': 'EfficientNetV2B0_FineTurning',#'EfficientNetV2B0_FineTurning',EfficientNetV2B0_FineTurning_real_dark
    'pretrain_model': EfficientNetV2B0,
    'classificationLayers': [Flatten(), Dense(len(classes), activation='softmax')],
    'weights': 'imagenet',
    'image_size': 224,
    'num_epochs': 10,
    'batch_training': 32,
    'batch_valid': 32,
    # learning rate
    'lr_schedule': True,
    'learning_rate': 0.001,
    # Adam params
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-07,
    # preprocessing params
    'probability_dark': 0.4,
    # fine-turing (trainable=True)
    'trainable': True,
    # call back function params
    'early_stop_patience': 10,
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

if __name__ == '__main__':
    params = get_model_parameters()
    train_directories = [PathDatasets.COCO2017_TRAIN.value]
    ratio = np.array([0, 20, 40, 60, 80, 100])
    do_train = False
    real_dark_train = False
    seed_aug = 0
    test_option = Test_options.Enhance

    enhance_methods = {'AHE': preproceessing_adaptive_histogram_equalization, 'HE': preprocessing_histogram_equalization}
    grid_search_params = {'lr_schedule': True, 'batch_training': [32, 64, 128],
                          'batch_valid': [32, 64, 128], 'ModelName': ['grid1', 'grid2', 'grid3']}
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
    if Test_options.GRID_SEARCH in test_option:
        ds_tests, ds_names = load_test_dataset(preprocess_input)
        max_size = max([len(v) for v in grid_search_params.values()])
        for i in range(max_size):
            params = edit_parameters_grid_search(grid_search_params, params, i)
            df_summery = run_one(params, do_train, train_directories, ds_tests, ds_names, df_summery)
            df_summery.to_excel(f"{params['ModelName']}.xlsx")
    if Test_options.Enhance in test_option:
        # do Enhance image
        for name_pre, preprocess in enhance_methods.items():
            ds_tests, ds_names = load_test_dataset(preprocess, only_dark=True)
            # create a CLAHE object (Arguments are optional).
            checkpoint_path, figure_path, \
            saved_models_path = get_directories_and_paths(params['ModelName']+ '_10_0')
            dp = load_saved_model(saved_models_path, params['ModelName'] + '_10_0', figure_path + '_' + name_pre)
            df_summery = evaluate_model_on_datasets(dp, df_summery, ds_tests, ds_names, params['ModelName'] + '_' + name_pre)
            df_summery.to_excel(params['ModelName'] + '_' + name_pre + '.xlsx')
        pass



