# https://keras.io/ko/applications/#vgg19 여길 꼭 들어가보기. 사실 keras에는 모든 모형들이 다 있음.

import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file, to_categorical
import sys


# VGG19 모형과 관련된 초기값이 저장되어 있는 링크. 접속시 바로 다운됨. FC층을 어떻게 처리할 건지에 따라 두가지 초기값들이 있음.
# 이 값은 ImageNet 데이터를 가지고 학습한 초기값임.
# 내가 쓸 모형은 이 초기값이 필요 없기 때문에, 내가 초기값을 뽑아내는 작업을 해야한다.
WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

# -*- coding: utf-8 -*-
'''VGG19 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''
('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
"""Instantiates the VGG19 architecture.
 Optionally loads weights pre-trained on ImageNet. Note that when using TensorFlow,
 for best performance you should set
 `image_data_format="channels_last"` in your Keras config
 at ~/.keras/keras.json.
 The model and the weights are compatible with both
 TensorFlow and Theano. The data format
 convention used by the model is the one
 specified in your Keras config file.

 # Arguments
     include_top: whether to include the 3 fully-connected
         layers at the top of the network.
     weights: one of `None` (random initialization)
         or "imagenet" (pre-training on ImageNet).
     input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
         to use as image input for the model.
     input_shape: optional shape tuple, only to be specified
         if `include_top` is False (otherwise the input shape
         has to be `(224, 224, 3)` (with `channels_last` data format)
         or `(3, 224, 244)` (with `channels_first` data format).
         It should have exactly 3 inputs channels,
         and width and height should be no smaller than 48.
         E.g. `(200, 200, 3)` would be one valid value.
     pooling: Optional pooling mode for feature extraction
         when `include_top` is `False`.
         - `None` means that the output of the model will be
             the 4D tensor output of the
             last convolutional layer.
         - `avg` means that global average pooling
             will be applied to the output of the
             last convolutional layer, and thus
             the output of the model will be a 2D tensor.
         - `max` means that global max pooling will
             be applied.
     classes: optional number of classes to classify images
         into, only to be specified if `include_top` is True, and
         if no `weights` argument is specified.
 # Returns
     A Keras model instance.
 """

#1. Original VGG19
def VGG19(input_shape=(224, 224, 1), classes=7, include_top=True, pooling=None, weights=None):

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        output = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            output = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            output = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(img_input, output, name='vgg19')

    # Load weights.
    if weights == 'imagenet': # 내 모형에서는 쓸모없다. 다만, 나중의 혹시모를 참고를 위해 코드는 남겨놓는다.
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH, cache_subdir='models', file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP, cache_subdir='models',
                                    file_hash='253f8cb515780f3b799900260a226db6')

        model.load_weights(weights_path)  # 경로에 있는 초기치 weights가져오기

    elif weights is not None:
        model.load_weights(weights)

    return model

model = VGG19(input_shape=(224, 224, 3), classes=7)

# 모수가 데이터에 비해 굉장히 많기 때문에 모수의 수를 좀 줄여서 확인해보자.
# 기본은 모형을 조절하는 것이 아닌, 데이터를 뻥튀기 하는 것임을 늘 잊지말기!!
model.summary()

