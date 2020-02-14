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


# 경로 확인
sys.path
sys.path.append("C:\\Users\\82104\\Desktop\\연구실\\2019_Winter\\My Project\\Code\\Class Module")

from lrn import LRN #만든 모듈, class

'''
아래 코드는 f1-score를 정의한 함수.
tensorflow2.0 버전 이후로 f1-score 지원이 metrics애서 사라졌다.
'''

def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)
    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    # return a single tensor value
    return recall

def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)
    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    # return a single tensor value
    return precision

def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())
    # return a single tensor value
    return _f1score


# 내 노트북이 아닌, 전산실 컴퓨터에서 돌렸으므로, 다시돌리려면 경로 수정할것!
os.chdir("C:\\Users\\82104\\Desktop\\fer2013\\mydata")


# 1. original alexnet

# data import
x_train = pd.read_csv("X_train.csv",header=0,index_col=0)
x_valid = pd.read_csv("X_private_test.csv",header=0,index_col=0)
x_test = pd.read_csv("X_public_test.csv",header=0,index_col=0)
y_train = pd.read_csv("y_train.csv",header=0,index_col=0)
y_valid = pd.read_csv("y_private_test.csv",header=0,index_col=0)
y_test = pd.read_csv("y_public_test.csv",header=0,index_col=0)


# data handling
# uint는 부호없는 정수로, 타입을 바꿔줘야함!
size = 224
x_train = np.array(x_train).reshape([-1,48,48,1])
x_train.shape
x_train_zoom = np.zeros([x_train.shape[0],size,size,1],dtype="float32")
for i in range(x_train.shape[0]):
    x_train_zoom[i,:] = cv2.resize(x_train[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,1) /255
x_train_zoom.shape

x_valid = np.array(x_valid).reshape([-1,48,48,1])
x_valid_zoom = np.zeros([x_valid.shape[0],size,size,1],dtype="float32")
for i in range(x_valid.shape[0]):
    x_valid_zoom[i,:] = cv2.resize(x_valid[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,1) /255
x_valid_zoom.shape

x_test = np.array(x_test).reshape([-1,48,48,1])
x_test_zoom = np.zeros([x_test.shape[0],size,size,1],dtype="float32")
for i in range(x_test.shape[0]):
    x_test_zoom[i,:] = cv2.resize(x_test[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,1) /255
x_test_zoom.shape


y_train = np.array(y_train).reshape([-1,])
y_valid = np.array(y_valid).reshape([-1,])
y_test = np.array(y_test).reshape([-1,])




# Alexnet을 최대한 논문에 가깝게 맞춰 모형작성.
# 또한, Data Augmentation은 컴퓨터 성능의 한계로 하지 않음.
# 출처 : https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py


def Alexnet(img_shape=(227, 227, 1), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape, strides=4,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer1_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3),strides=2))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same',strides=1,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer2_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3),strides=2))

	# Layer 3
	alexnet.add(Conv2D(384, (3, 3), padding='same',
					   strides=1,kernel_regularizer=l2(l2_reg)))
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
	alexnet.add(MaxPooling2D(pool_size=(3, 3),strides=2))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(4096,kernel_regularizer=l2(l2_reg)))

	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

#내 데이터 맞춤형 모형
model = Alexnet(img_shape=(227, 227, 1), n_classes=7, l2_reg=0.,weights=None)

#모수가 데이터에 비해 굉장히 많지만, 일단 진행
model.summary()

# 여기는 학습의 효율을 위해. Adam으로 넘어간다.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy',
              metrics=['accuracy',f1score])
model.fit(x_train_zoom,y_train,batch_size=128, validation_data=(x_valid_zoom,y_valid) , epochs=2)

# 굉장히 performance가 저조하다. 모수가 데이터에 비해 너무나도 많음.


# 2. my alexnet
# Alexnet을 개조
# Data Augmentation은 컴퓨터 성능의 한계로 못하기 때문에 변형함.


# data import
x_train = pd.read_csv("X_train.csv",header=0,index_col=0)
x_valid = pd.read_csv("X_private_test.csv",header=0,index_col=0)
x_test = pd.read_csv("X_public_test.csv",header=0,index_col=0)
y_train = pd.read_csv("y_train.csv",header=0,index_col=0)
y_valid = pd.read_csv("y_private_test.csv",header=0,index_col=0)
y_test = pd.read_csv("y_public_test.csv",header=0,index_col=0)


# data handling
# size 를 55로 두는이유는, original alexnet의 첫번째 conv. 이후에 feature map 사이즈가 55이므로, 코드를 쉽게보기 위함.
size = 57
x_train = np.array(x_train).reshape([-1,48,48,1])
x_train.shape
x_train_zoom = np.zeros([x_train.shape[0],size,size,1],dtype="float32")
for i in range(x_train.shape[0]):
    x_train_zoom[i,:] = cv2.resize(x_train[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,1) /255
x_train_zoom.shape

x_valid = np.array(x_valid).reshape([-1,48,48,1])
x_valid_zoom = np.zeros([x_valid.shape[0],size,size,1],dtype="float32")
for i in range(x_valid.shape[0]):
    x_valid_zoom[i,:] = cv2.resize(x_valid[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,1) /255
x_valid_zoom.shape

x_test = np.array(x_test).reshape([-1,48,48,1])
x_test_zoom = np.zeros([x_test.shape[0],size,size,1],dtype="float32")
for i in range(x_test.shape[0]):
    x_test_zoom[i,:] = cv2.resize(x_test[i,:].astype('uint8'), (size, size),
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,1) /255
x_test_zoom.shape




# 다음의 절차로 모형을 개조한다.

# 1. 227의 1/4 연산인 57로 이미지사이즈를 재조정한다.
# 2. 모수와 관련이 가장 깊은 fc층에서, 기존의 4096개의 노드를 1/16 (비율) 배 만큼, 즉 256개로줄인다.
# 3. 다음과 같이 모형을 재구성한다.

# convolution layer
# 입력 : 57x57x1
# 첫번째 층 : 5x5 필터 24장, strides = 2 -> maxpooling 3x3 , stirdes = 2   ===> 13 x 13 x 24 feature map
# 두번째 층 : 3x3 필터 64장, strides = 1, padding = "same"                 ===> 13 x 13 x 64 feature map
# 세번째 층 : 3x3 필터 96장, strides = 1                                   ===> 11 x 11 x 96 feature map
# 네번째 층 : 3x3 필터 96장, strides = 1, padding = "same"                 ===> 11 x 11 x 96 feature map
# 다섯째 층 : 3x3 필터 64장, strides = 1 -> maxpooling 3x3 , strides = 2   ===> 4 x 4 x 64 feature map

# fc layer
# 여섯째 층 : 노드 256개, dropout = 0.5
# 일곱째 층 : 노드 256개, dropout = 0.5
# 여덟째 층 : 노드 7개



def Alexnet(img_shape=(57, 57, 1), n_classes=10, l2_reg=0.,weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(24, (5, 5), input_shape=img_shape, strides=2,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer1_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3),strides=2))

	# Layer 2
	alexnet.add(Conv2D(64, (3, 3), padding='same',strides=1,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer2_LRN'))

	# Layer 3
	alexnet.add(Conv2D(96, (3, 3),  strides=1,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer3_LRN'))

	# Layer 4
	alexnet.add(Conv2D(96, (3, 3), padding='same'))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer4_LRN'))

	# Layer 5
	alexnet.add(Conv2D(64, (3, 3)))
	alexnet.add(Activation('relu'))
	alexnet.add(LRN(name='layer5_LRN'))
	alexnet.add(MaxPooling2D(pool_size=(3, 3),strides=2))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(256,kernel_regularizer=l2(l2_reg)))

	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(256,kernel_regularizer=l2(l2_reg)))
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

#내 데이터 맞춤형 모형
model = Alexnet(img_shape=(57, 57, 1), n_classes=7, l2_reg=0.,weights=None)

#모수가 데이터에 비해 굉장히 많지만, 일단 진행
model.summary()

# 여기는 학습의 효율을 위해. Adam으로 넘어간다.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy',
              metrics=['accuracy',f1score])
model.fit(x_train_zoom,y_train,batch_size=128, validation_data=(x_valid_zoom,y_valid) , epochs=2)
