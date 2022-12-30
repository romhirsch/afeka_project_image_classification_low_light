
import os
import pathlib
import time
import datetime
import imageio
from glob import glob



import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


from matplotlib import pyplot as plt
from IPython import display
from termcolor import colored
from tqdm import tqdm
from IPython.display import Image
import PIL
from PIL import ImageDraw
from IPython import display

def color_print(print_str,
                print_color='green'):

    '''print in given  color (default green)'''
    print(colored(print_str ,print_color))


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f'setting seed to {seed}')


class CFG:
    # Dimension of image
    IMG_WIDTH = 512
    IMG_HEIGHT = 512

    # resize image
    resize_height = 700
    resize_width = 1200

    # the lambda param in loss
    LAMBDA = 10
    # --------train pipe-------------
    BUFFER_SIZE = 100
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 2

    # cache
    cache = 50

    # learning rate
    learning_rate = 0.00025

    seed = 7


set_seed(CFG.seed)

# Data directories
day_dir = 'day'
night_dir = 'night'


#plotting a sample image
plt.figure(figsize=(16,8))

img = plt.imread(day_dir + '/' + os.listdir(day_dir)[0])
plt.imshow(img)
plt.axis('off')
plt.title('sample image')
print(f'Image dimensions {img.shape}')

"""
 building data loading pipeline
"""


def load_image(image_file):
    '''load a image file'''
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    return image


def random_crop(image):
    '''randomly crop image into defined size '''
    cropped_image = tf.image.random_crop(image, size=[CFG.IMG_HEIGHT, CFG.IMG_WIDTH, 3])

    return cropped_image


def normalize(image):
    '''normalizing the images to [-1, 1]'''
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def de_normalize(image):
    '''De normalize the image to be in range (0,1)'''

    return (image * 0.5) + 0.5


def image_augmentations(image):
    '''perform spatial augmentations (rotation and flips) on input image

    from : https://www.kaggle.com/code/dimitreoliveira/improving-cyclegan-monet-paintings'''

    # --------------------rotations----------
    # rotation probabliity
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if p_rotate > .8:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .6:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .4:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # ----------------------Flips---------------------
    p_flip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if p_flip > 0.7:
        image = tf.image.random_flip_left_right(image)
    elif p_flip < 0.3:
        image = tf.image.random_flip_up_down(image)

    return image


def random_jitter(image):
    '''resize and randommly crop the input image'''

    #     # resizing image
    image = tf.image.resize(image, size=(CFG.resize_height, CFG.resize_width),
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 512,512
    image = random_crop(image)

    return image


def preprocess_image_train(image):
    image = load_image(image)
    image = random_jitter(image)
    image = image_augmentations(image)
    image = normalize(image)
    return image


# same function, withou the augemntation
def preprocess_image_eval(image):
    image = load_image(image)
    image = random_jitter(image)
    image = normalize(image)
    return image


def create_img_dataset(directory,
                       image_preprocess_fn,
                       image_extension='jpg',
                       repeat=True
                       ):
    '''create a tf dataset object from a directory of images'''
    img_list = glob(directory + f'/*{image_extension}')

    dataset = tf.data.Dataset.list_files(img_list)

    dataset = dataset.map(image_preprocess_fn,
                          num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.shuffle(CFG.BUFFER_SIZE)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    return dataset


Day_Dataset = create_img_dataset(directory=day_dir, image_preprocess_fn=preprocess_image_train)

# without augmentation
Day_eval = create_img_dataset(directory=day_dir,
                              image_preprocess_fn=preprocess_image_eval)

fig, ax = plt.subplots(figsize=(16, 8))

inp_img = next(iter(Day_Dataset))
plt.imshow(de_normalize(inp_img[0]))
plt.title('Sample Day image')
plt.axis('off')


Night_Dataset = create_img_dataset(directory = night_dir,image_preprocess_fn = preprocess_image_train)
Night_eval = create_img_dataset(directory = night_dir,
                                image_preprocess_fn = preprocess_image_eval)
fig,ax = plt.subplots(figsize=(16,8))


inp_img = next(iter(Night_Dataset))
plt.imshow(de_normalize(inp_img[0]))
plt.title('Sample Night image')
plt.axis('off')



# train data set

Train_Dataset = tf.data.Dataset.zip((Day_Dataset, Night_Dataset))

"""
Building Model
"""

# conv weights initilaizer
conv_initializer = tf.random_normal_initializer(mean=0.0,
                                                stddev=0.02)

# init for intance normalization
gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                       stddev=0.02)


def downsample(input_layer,
               filters,
               name,
               size=3,
               strides=2,
               activation=tf.keras.layers.ReLU(),
               ):
    '''perform a downsampling by applying a convolution,followed by instance norm and activation'''
    conv = tf.keras.layers.Conv2D(filters,
                                  size,
                                  strides=strides,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=conv_initializer,
                                  name=f'encoder_{name}')(input_layer)

    conv = tfa.layers.InstanceNormalization(axis=-1, gamma_initializer=gamma_initializer)(conv)

    conv = activation(conv)

    return conv


def upsample(input_layer,
             filters,
             name,
             size=3,
             strides=2,
             activation='relu'):
    res = tf.keras.layers.Conv2DTranspose(filters, size,
                                          strides=strides,
                                          padding='same',
                                          use_bias=False,
                                          kernel_initializer=conv_initializer,
                                          name=f'decoder_{name}')(input_layer)

    res = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(res)

    res = tf.keras.layers.Activation(activation)(res)

    return res


def residual_block(input_layer,
                   size=3,
                   strides=1,
                   name='block_x'):
    '''performs 2 convolutions followed by an added skip connection with the input'''

    filters = input_layer.shape[-1]
    block = tf.keras.layers.Conv2D(filters,
                                   size,
                                   strides=strides,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=conv_initializer,
                                   name=f'residual_{name}')(input_layer)

    block = tf.keras.layers.Activation('relu')(block)
    block = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False,
                                   kernel_initializer=conv_initializer, name=f'transformer_{name}_2')(block)

    # skip connection
    res = tf.keras.layers.Add()([block, input_layer])

    return res


def concat_layer(layer_1, layer_2, name):
    '''concatenation of layers for skip connections'''
    return tf.keras.layers.Concatenate(name=name)([layer_1, layer_2])

"""
Building Generator
"""


def get_generator(num_residual_connections=6):
    input_ = tf.keras.layers.Input(shape=(CFG.IMG_WIDTH, CFG.IMG_HEIGHT, 3),
                                   name='input_layer')

    # -----------------------ENCODER-------------------------------
    # downsample images
    enc1 = downsample(input_layer=input_, filters=64, strides=1, size=7, name='dwn_1')  # (bs, 512,512, 64)
    enc2 = downsample(input_layer=enc1, filters=128, size=3, strides=2, name='dwn_2')  # (bs, 256, 256, 128)
    enc3 = downsample(input_layer=enc2, filters=256, size=3, strides=2, name='dwn_3')  # (bs, 128,128,256)
    enc4 = downsample(input_layer=enc3, filters=256, size=3, strides=2, name='dwn_4')  # (bs, 64,64,256)

    # -----------------------Residual connections-------------------------------
    x = enc4
    for n in range(num_residual_connections):
        x = residual_block(input_layer=x, name=f'res_block_{n + 1}')  # (bs, 64, 64, 256)

    # -----------------------DECODER-------------------------------
    # UNET like skip connection
    # upsample 1
    x_skip = concat_layer(layer_1=x, layer_2=enc4, name='skip_1')
    dec1 = upsample(x_skip, filters=256, name='upsam_1')  # (bs, 128, 128, 256)

    # upsample 2
    x_skip = concat_layer(layer_1=dec1, layer_2=enc3, name='skip_2')
    dec_2 = upsample(x_skip, filters=128, name='upsam_2')  # (bs, 256, 256, 128)

    # upsample 3
    x_skip = concat_layer(layer_1=dec_2, layer_2=enc2, name='skip_3')
    dec_3 = upsample(x_skip, filters=64, name='upsam_3')  # (bs, 512, 512, 64)

    # penultimate
    x_skip = concat_layer(layer_1=dec_3,
                          layer_2=enc1,
                          name='skip_final')

    output = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same',
                                    kernel_initializer=conv_initializer, use_bias=False, activation='tanh',
                                    name='output_layer')(x_skip)

    return tf.keras.models.Model(inputs=input_, outputs=output)


# day images -> night images
day2night_gen = get_generator()

# night images -> day images
night2day_gen = get_generator()

# plot model
# tf.keras.utils.plot_model(day2night_gen)
#passing in a input to generator for check
#plot a sample output
gen_output = night2day_gen(inp_img, training=False)[0]
plt.subplots(1,2,figsize=(16,8))


plt.subplot(1,2,1)
plt.imshow(gen_output.numpy().squeeze())
plt.title('Untrained Night2Day Generator output')
plt.axis('off')


plt.subplot(1,2,2)
plt.imshow(de_normalize(inp_img[0]))
plt.title('Original Night image')
plt.axis('off')

"""
Building Discriminator
"""


def PATCH_discriminator(leak_rate=0.2):
    '''PATCH discriminator network'''
    leaky_relu = tf.keras.layers.LeakyReLU(leak_rate)

    input_ = tf.keras.layers.Input(shape=(CFG.IMG_WIDTH, CFG.IMG_HEIGHT, 3),
                                   name='input_layer')
    # Encoder
    # Input image 512,512
    x = downsample(input_layer=input_, filters=64, strides=2, size=4, name='dwn_1', activation=leaky_relu)  # h,w =256
    x = downsample(input_layer=x, filters=128, strides=2, size=4, name='dwn_2', activation=leaky_relu)  # h,w =128
    x = downsample(input_layer=x, filters=256, strides=2, size=4, name='dwn_3', activation=leaky_relu)  # h,w = 64
    x = downsample(input_layer=x, filters=512, strides=2, size=4, name='dwn_4', activation=leaky_relu)  # h,w = 32
    x = downsample(input_layer=x, filters=512, strides=1, size=4, name='dwn_5', activation=leaky_relu)  # h,w = 32

    output = tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=conv_initializer)(
        x)  # (29, 29, 1)

    return tf.keras.models.Model(inputs=input_, outputs=output)


# create instance of discriminators
day2night_disc = PATCH_discriminator()  # identify night images
night2day_disc = PATCH_discriminator()  # identify day images
#check on dicriminator

disc_output = day2night_disc(inp_img, training=False)
plt.subplots(1,1,figsize=(8,8))

plt.imshow(disc_output.numpy().mean(axis=0),cmap='gray')
plt.title('Untrained Night2Day disc output')
plt.axis('off')



def generate_cycle(gen_1, gen_2, input_image):
    '''generate a full cycle of images using given generators'''
    gen_img_1 = gen_1(input_image, training=True)
    gen_img_2 = gen_2(gen_img_1, training=True)

    return gen_img_1, gen_img_2


def calc_and_apply_gradients(tape,
                             model,
                             loss,
                             optimizer):
    '''Apply gradients for a given model using given optimizer'''

    # calculate gradients of loss function
    gradients = tape.gradient(loss, model.trainable_variables)

    # apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return

"""
loss
"""


def discriminator_loss(real, generated):
    '''discriminator Binary CrossEntropy loss'''
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


# Generator Adverserial loss
def generator_loss(generated):
    '''adverserial generator loss (BCE)'''
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated), generated)


# Cycle consistency loss

def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    '''pixel wise cycle loss between original image and cycled image'''
    mae_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * mae_loss


# identity loss
def identity_loss(real_image, same_image, LAMBDA):
    mae_loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * mae_loss

"""
Cycle GAN
"""


class CycleGAN(tf.keras.models.Model):
    def __init__(self,
                 lambda_cycle=10):
        super(CycleGAN, self).__init__()
        self.gen_d2n = day2night_gen  # Day -> Night
        self.gen_n2d = night2day_gen  # Night -> Day
        self.disc_d2n = day2night_disc  # Classifies Night Images
        self.disc_n2d = night2day_disc  # Classifier Day Images
        self.lambda_cycle = lambda_cycle  # lambda in cycle consistancy loss

    def compile(self,
                gen_loss_fn,
                disc_loss_fn,
                cycle_loss_fn,
                identity_loss_fn,
                common_opt=tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate, beta_1=0.5)):
        super(CycleGAN, self).compile()

        # -------optimizers ---------
        self.opt_gen_d2n = common_opt
        self.opt_gen_n2d = common_opt
        self.opt_disc_d2n = common_opt
        self.opt_disc_n2d = common_opt

        # -------losses ---------
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        day_image, night_image = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # -----day->night->day
            fake_night, cycled_day = generate_cycle(self.gen_d2n,
                                                    self.gen_n2d,
                                                    day_image)

            # --------night -> day- > night
            fake_day, cycled_night = generate_cycle(self.gen_n2d,
                                                    self.gen_d2n,
                                                    night_image)

            # ---------- generating itself (for identity loss)
            iden_day = self.gen_d2n(night_image, training=True)
            iden_night = self.gen_n2d(day_image, training=True)

            # -----------discriminator on real images
            disc_night = self.disc_d2n(night_image, training=True)
            disc_day = self.disc_n2d(day_image, training=True)

            # -----------discriminator on fake images-----------------
            disc_fake_night = self.disc_d2n(fake_night, training=True)
            disc_fake_day = self.disc_n2d(fake_day, training=True)

            # -------------------------generator loss-------------
            # ---1)adverserial loss
            night_gen_loss = self.gen_loss_fn(disc_fake_night)
            day_gen_loss = self.gen_loss_fn(disc_fake_day)

            # ---2)Cycle loss loss
            total_cycle_loss = self.cycle_loss_fn(night_image, cycled_night, self.lambda_cycle) + self.cycle_loss_fn(
                day_image, cycled_day, self.lambda_cycle)

            # +++++3) Total Gen loss (day gen and night gen)
            total_gen_d2n_loss = night_gen_loss + total_cycle_loss + self.identity_loss_fn(night_image, iden_night,
                                                                                           self.lambda_cycle)
            total_gen_n2d_loss = day_gen_loss + total_cycle_loss + self.identity_loss_fn(day_image, iden_day,
                                                                                         self.lambda_cycle)

            # -------------------------Discriminator loss-------------
            night_disc_loss = self.disc_loss_fn(disc_night,
                                                disc_fake_night)  # check classifying generated and real night
            day_disc_loss = self.disc_loss_fn(disc_day, disc_fake_day)  # check  classifying generated and real day

        ## ------------------------- Calculating and Updating gradients------------------

        # day->night gen gradeints
        _ = calc_and_apply_gradients(tape=tape,
                                     model=self.gen_d2n,
                                     loss=total_gen_d2n_loss,
                                     optimizer=self.opt_gen_d2n)

        # night - >day  gen gradeints
        _ = calc_and_apply_gradients(tape=tape,
                                     model=self.gen_n2d,
                                     loss=total_gen_n2d_loss,
                                     optimizer=self.opt_gen_n2d)

        #  discrim gradients (classifies night images)
        _ = calc_and_apply_gradients(tape=tape,
                                     model=self.disc_d2n,
                                     loss=night_disc_loss,
                                     optimizer=self.opt_disc_d2n)

        # Day discrim gradients (classifies day images)
        _ = calc_and_apply_gradients(tape=tape,
                                     model=self.disc_n2d,
                                     loss=day_disc_loss,
                                     optimizer=self.opt_disc_n2d)

        return {'gen_D2N_loss': total_gen_d2n_loss,
                'gen_N2D_loss': total_gen_n2d_loss,
                'disc_day_loss': day_disc_loss,
                'disc_night_loss': night_disc_loss
                }

#creat a instance of Cycle gan
gan = CycleGAN()


#complie with the losses
gan.compile(gen_loss_fn=generator_loss,
            disc_loss_fn=discriminator_loss,
            cycle_loss_fn=calc_cycle_loss,
            identity_loss_fn=identity_loss)


# learning rate schedule

def scheduler(epoch,
              lr,
              decay_rate=0.05,
              warm_up_period=10):
    if epoch < warm_up_period:
        return lr
    elif (epoch > warm_up_period and epoch < 40):
        return lr * tf.math.exp(decay_rate)
    else:
        return lr * tf.math.exp(decay_rate * 2)


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler,
                                                        verbose=0)

# early stopping

# early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'gen_N2D_loss',
#                                               mode = 'min',
#                                               patience = 10,
#                                              restore_best_weights = True)

# early stopping : from https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):

        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.n2d_loss = np.Inf
        self.d2n_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        n2d_loss = logs.get('gen_N2D_loss')
        d2n_loss = logs.get('gen_D2N_loss')

        # If both the conditions are met, continue training
        if (np.less(n2d_loss, self.n2d_loss) and np.less(d2n_loss, self.d2n_loss)):
            self.d2n_loss = d2n_loss
            self.n2d_loss = n2d_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

        # if above xondition not met, continue training till patiance epochs
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# --------------------------------------Viz Callbacks : from https://www.kaggle.com/code/dimitreoliveira/improving-cyclegan-monet-paintings -----------------------------------------
def display_samples(ds, n_samples):
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        example_sample = next(ds_iter)
        plt.subplot(121)
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()


def display_generated_samples(ds, model, n_samples):
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        example_sample = next(ds_iter)
        generated_sample = model.predict(example_sample)

        f = plt.figure(figsize=(16, 8))

        plt.subplot(121)
        plt.title('Input image')
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis('off')

        plt.subplot(122)
        plt.title('Generated image')
        plt.imshow(generated_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()


def evaluate_cycle(ds, generator_a, generator_b, n_samples=1):
    fig, axes = plt.subplots(n_samples, 3, figsize=(22, (n_samples * 6)))
    axes = axes.flatten()

    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        idx = n_sample * 3
        example_sample = next(ds_iter)
        generated_a_sample = generator_a.predict(example_sample)
        generated_b_sample = generator_b.predict(generated_a_sample)

        axes[idx].set_title('Input image', fontsize=18)
        axes[idx].imshow(example_sample[0] * 0.5 + 0.5)
        axes[idx].axis('off')

        axes[idx + 1].set_title('Generated image', fontsize=18)
        axes[idx + 1].imshow(generated_a_sample[0] * 0.5 + 0.5)
        axes[idx + 1].axis('off')

        axes[idx + 2].set_title('Cycled image', fontsize=18)
        axes[idx + 2].imshow(generated_b_sample[0] * 0.5 + 0.5)
        axes[idx + 2].axis('off')

    plt.show()


def predict_and_save(input_ds, generator_model, output_path):
    i = 1
    for img in input_ds:
        prediction = generator_model(img, training=False)[0].numpy()  # make predition
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)  # re-scale
        im = PIL.Image.fromarray(prediction)
        im.save(f'{output_path}{str(i)}.jpg')
        i += 1


# Callback
class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self,
                 num_img=1,
                 day_paths='generated_day',
                 night_paths='generated_night'):
        self.num_img = num_img
        self.day_paths = day_paths
        self.night_paths = night_paths

        # dir to save genereated day images
        if not os.path.exists(self.day_paths):
            os.makedirs(self.day_paths)

        # dir to save genereated night images
        if not os.path.exists(self.night_paths):
            os.makedirs(self.night_paths)

    def on_epoch_end(self, epoch, logs=None):
        # generated night
        for i, img in enumerate(Day_eval.take(self.num_img)):
            prediction = day2night_gen(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f'{self.night_paths}/generated_{i}_{epoch + 1}.png')

        # generated day images
        for i, img in enumerate(Night_eval.take(self.num_img)):
            prediction = night2day_gen(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f'{self.day_paths}/generated_{i}_{epoch + 1}.png')

EPOCHS = 75
callbacks = [lr_scheduler,GANMonitor(),CustomEarlyStopping(patience = 10)]
steps_per_epoch = 200


history = gan.fit(Train_Dataset,
                epochs = EPOCHS,
                steps_per_epoch=steps_per_epoch,
                callbacks = callbacks)


# from https://www.kaggle.com/code/dimitreoliveira/improving-cyclegan-monet-paintings
def create_gif(images_path, gif_path):
    images = []
    filenames = glob(images_path)
    filenames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for epoch, filename in enumerate(filenames):
        img = PIL.ImageDraw.Image.open(filename)
        ImageDraw.Draw(img).text((0, 0),  # Coordinates
                                 f'Epoch {epoch + 1}')
        images.append(img)
    imageio.mimsave(gif_path, images, fps=2)  # Save gif


create_gif('./generated_day/*.png', 'day.gif')

print('Training progress for Night -> Day Generator')
display.Image('./day.gif')
create_gif('./generated_night/*.png', 'night.gif')


print('Training progress for Day-> Generator')
display.Image('./night.gif')
display_generated_samples(Day_eval.take(2), day2night_gen, 2)