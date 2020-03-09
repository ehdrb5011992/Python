#pip install tensorflow==2.1.0

import sys
# 모듈로 받을 경로 확인
sys.path
# 내 노트북이 아닌, 전산실 컴퓨터의 colab에서 돌렸으므로, 다시돌리려면 경로 수정할것!
sys.path.append("/content/drive/My Drive/Colab Notebooks/project")
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

from f1score import macro_f1score, weighted_f1score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.__version__
tf.test.gpu_device_name()


# data import
x_train = pd.read_csv("mydata/X_train.csv",header=0,index_col=0)
x_valid = pd.read_csv("mydata/X_private_test.csv",header=0,index_col=0)
x_test = pd.read_csv("mydata/X_public_test.csv",header=0,index_col=0)
y_train = pd.read_csv("mydata/y_train.csv",header=0,index_col=0)
y_valid = pd.read_csv("mydata/y_private_test.csv",header=0,index_col=0)
y_test = pd.read_csv("mydata/y_public_test.csv",header=0,index_col=0)


# data handling
x_train = np.array(x_train).reshape([-1,48,48,3])
x_valid = np.array(x_valid).reshape([-1,48,48,3])
x_test = np.array(x_test).reshape([-1,48,48,3])

y_train=to_categorical(y_train) # one hot encoding
y_valid=to_categorical(y_valid)
y_test=to_categorical(y_test)

# data handling
size = 64 #적당한 크기로 잡음.
x_train = np.array(x_train).reshape([-1,48,48,3])

x_train_zoom = np.zeros([x_train.shape[0],size,size,3],dtype="float32")

for i in range(x_train.shape[0]):
    x_train_zoom[i,:] = cv2.resize(x_train[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,3) /255

x_train = x_train / 255


x_valid = np.array(x_valid).reshape([-1,48,48,3])
x_valid_zoom = np.zeros([x_valid.shape[0],size,size,3],dtype="float32")
for i in range(x_valid.shape[0]):
    x_valid_zoom[i,:] = cv2.resize(x_valid[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,3) /255


 x_valid = x_valid / 255

x_test = np.array(x_test).reshape([-1,48,48,3])
x_test_zoom = np.zeros([x_test.shape[0],size,size,3],dtype="float32")
for i in range(x_test.shape[0]):
    x_test_zoom[i,:] = cv2.resize(x_test[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,3) /255

x_test = x_test / 255


datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=[0.9,1.0])

datagen_val = ImageDataGenerator()

xy_valid_zoom_gen = datagen_val.flow(x_valid_zoom,y_valid,batch_size=128)
xy_valid_gen = datagen_val.flow(x_valid,y_valid,batch_size=128)



# VGG19 모형과 관련된 초기값이 저장되어 있는 링크. 접속시 바로 다운됨. FC층을 어떻게 처리할 건지에 따라 두가지 초기값들이 있음.
# 이 값은 ImageNet 데이터를 가지고 학습한 초기값임.
# 내가 쓸 모형은 이 초기값이 필요 없기 때문에, 내가 초기값을 뽑아내는 작업을 해야한다.

# 다시한번 얘기하지만, 아래의 두 변수는 내가 아래의 코드에서 쓸 변수는 아님. 나중의 혹시모를 상황을 대비해 남겨놓음.
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



# VGG19를 최대한 논문에 가깝게 맞춰 모형작성.
# 또한, Data Augmentation은 컴퓨터 성능의 한계로 하지 않음.

def VGG19(input_shape=(224, 224, 3), classes=7, include_top=True, pooling=None, weights=None):

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


# 여기는 학습의 효율을 위해. Adam으로 넘어간다.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy',macro_f1score,weighted_f1score])

early_stopping = EarlyStopping(monitor = 'val_macro_f1score',patience = 3 , verbose=1,mode='max')

model.fit(datagen.flow(x_train_zoom,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_zoom_gen, validation_steps=len(x_valid_zoom)/128, epochs=100, callbacks=[early_stopping])

_, acc, mac_f1, wei_f1 = model.evaluate(x_test_zoom,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))


# 한층당 W 와 b , 2개씩 있으므로 11개층이라면 22개의 모수 벡터 및 행렬이 출력된다.
W = model.get_weights()

W

model.weights # 참고. get_weights()와 weights는 다른 매서드임을 알자.

# 형태는 이렇게 생김
np.array(W).shape





