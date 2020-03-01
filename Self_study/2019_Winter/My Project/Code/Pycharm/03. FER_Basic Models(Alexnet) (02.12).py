import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
import sys


# Alexnet을 최대한 논문에 가깝게 맞춰 모형작성.
# 또한, Data Augmentation은 컴퓨터 성능의 한계로 하지 않음.
# 출처 : https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py

def Alexnet(img_shape=(227, 227, 3), n_classes=10, l2_reg=0., weights=None):
	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape, strides=4, kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer1_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=2))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same', strides=1, kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer2_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=2))

	# Layer 3
	alexnet.add(Conv2D(384, (3, 3), padding='same', strides=1, kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer3_LRN'))

	# Layer 4
	alexnet.add(Conv2D(384, (3, 3), padding='same'))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer4_LRN'))

	# Layer 5
	alexnet.add(Conv2D(256, (3, 3), padding='same'))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer5_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=2))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(4096, kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096, kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

model = Alexnet(img_shape=(227, 227, 3), n_classes=7, l2_reg=0.,weights=None)

#모수가 데이터에 비해 굉장히 많지만, 일단 진행
model.summary()

