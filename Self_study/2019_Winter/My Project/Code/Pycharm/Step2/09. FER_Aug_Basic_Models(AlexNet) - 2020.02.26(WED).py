#pip install tensorflow==2.1.0

import sys
# 모듈로 받을 경로 확인
sys.path
# 내 노트북이 아닌, 전산실 컴퓨터의 colab에서 돌렸으므로, 다시돌리려면 경로 수정할것!
sys.path.append("/content/drive/My Drive/Colab Notebooks/project")
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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

# 여기는 학습의 효율을 위해. Adam으로 넘어간다.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy',macro_f1score,weighted_f1score])
early_stopping = EarlyStopping(monitor='val_macro_f1score',patience=3,verbose=1,mode='max')

# early stopping
model.fit(datagen.flow(x_train_zoom,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_zoom_gen, validation_steps=len(x_valid_zoom)/128, epochs=100, callbacks=[early_stopping])

_, acc, mac_f1, wei_f1 = model.evaluate(x_test_zoom,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))
