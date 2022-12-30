import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
mnist = tf.keras.datasets.mnist
#
# (ds_train, ds_test), ds_info = tfds.load(
#     'coco',
#     split=['train', 'validation'],
#     with_info=True,
# )caltech101 sun397
x = tfds.as_numpy(tfds.load(
    'coco',
    split='train',
    as_supervised=True
))
x = 5