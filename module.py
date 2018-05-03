"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

import utils

############################################################
#  Utility Functions
############################################################


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


############################################################
#  DarkNet Graph
############################################################


def residual_block(input_tensor, kernel_sizes, num_filters, stage, block, use_bias=True):

    # TODO: Check if need batch normalization here?

    kernel_size1, kernel_size2 = kernel_sizes
    num_filter1, num_filter2 = num_filters

    conv_name_base = "res_" + str(stage) + "_" + str(block) + "_conv_"
    # bn_name_base = "res_" + str(stage) + "_" + str(block) + "_bn_"

    # Check if here need to add activation layer
    x = KL.Conv2D(num_filter1, (kernel_size1, kernel_size1), padding="SAME", name=conv_name_base + 'a',
                  use_bias=use_bias)(input_tensor)
    # x = BatchNorm(axis=3, name=bn_name_base + 'a')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    x = KL.Conv2D(num_filter2, (kernel_size2, kernel_size2), padding="SAME", name=conv_name_base + 'b',
                  use_bias=use_bias)(x)
    # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    x = KL.Add()([x, input_tensor])
    x = KL.LeakyReLU(alpha=0.1, name='res' + str(stage) + "_" + str(block) + '_out')(x)

    return x


def darknet_graph(input_image, architecture):
    assert architecture in ['darknet53']
    # input: 256x256
    x = KL.Conv2D(32, (3, 3), strides=(1, 1), padding="SAME", name='input_conv1', use_bias=True)(input_image)
    x = KL.LeakyReLU(alpha=0.1)(x)
    x = KL.Conv2D(64, (3, 3), strides=(2, 2), padding="SAME", name='input_conv2', use_bias=True)(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # stage1
    for b in range(1):
        x = residual_block(x, [1, 3], [32, 64], 1, b)
    S1 = x

    # stage2
    x = KL.Conv2D(128, (3, 3), strides=(2, 2), padding="SAME", name='s1_s2_conv1', use_bias=True)(x)
    for b in range(2):
        x = residual_block(x, [1, 3], [64, 128], 2, b)
    S2 = x

    # stage3
    x = KL.Conv2D(256, (3, 3), strides=(2, 2), padding="SAME", name='s2_s3_conv1', use_bias=True)(x)
    for b in range(8):
        x = residual_block(x, [1, 3], [128, 256], 3, b)
    S3 = x

    # stage4
    x = KL.Conv2D(512, (3, 3), strides=(2, 2), padding="SAME", name='s3_s4_conv1', use_bias=True)(x)
    for b in range(8):
        x = residual_block(x, [1, 3], [256, 512], 4, b)
    S4 = x

    # stage5
    x = KL.Conv2D(1024, (3, 3), strides=(2, 2), padding="SAME", name='s4_s5_conv1', use_bias=True)(x)
    for b in range(4):
        x = residual_block(x, [1, 3], [512, 1024], 5, b)
    S5 = x

    return S1, S2, S3, S4, S5


#####################################################################
# Data Generator
#####################################################################
def data_generator():
    pass

