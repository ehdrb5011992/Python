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

size=64
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
                                  interpolation=cv2.INTER_CUBIC).reshape(size,size,3) / 255

x_test = x_test / 255

# data argumentation

# Data Argumentation은 다음과 같은 옵션을 주어 실행한다. 이 생성방식은 모든 데이터에 적용.
# 1. 좌우 회전 10도 - 회전하다가 손실되는 데이터 방지 (작은 값)
# 2. 좌우 이동 10% (작은 값)
# 3. 좌우 반전 (표정 데이터이기 때문에 시도했으며, 상하반전은 시도하지 않았다.)
# 4. 줌범위 90% ~ 100% (너무 큰 줌은 데이터 손실을 유발함.)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=[0.9,1.0])

# train // training data 만 argumentation을 한다.
# featurewise 기능이나, ZCA whitening을 할 때 사용함. 그 외에는 굳이 선언할 필요 없음.
# 여기서 할 필요는 없지만, 주석처리는 안하고 넘어감.
datagen.fit(x_train)
datagen.fit(x_train_zoom)

# flow_from_directory를 사용하면 더 쉽게 변수를 저장할 수 있지만
# 우리는 작업해야 하는 데이터가 있으므로 위 메서드를 사용하지 않음.

# validation 은 그 자체로만 둔다.
# 다만 type은 gernerator로 바꿔준다.
datagen_val = ImageDataGenerator()

xy_valid_zoom_gen = datagen_val.flow(x_valid_zoom,y_valid,batch_size=128)
xy_valid_gen = datagen_val.flow(x_valid,y_valid,batch_size=128)

###################### 1. model - MLP ######################
#1) size = 64

# MLP model
inputs = tf.keras.Input(shape=(64,64,3))
x=tf.keras.layers.Flatten()(inputs)
x=tf.keras.layers.Dense(units=128,activation = 'relu',name='d1')(x)
x=tf.keras.layers.Dropout(0.3)(x)
x=tf.keras.layers.Dense(units=512,activation = 'relu',name='d2')(x)
x=tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(units=7,activation = tf.nn.softmax,name='d3')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',metrics=['accuracy',macro_f1score,weighted_f1score])
early_stopping = EarlyStopping(monitor='val_macro_f1score', patience=3, verbose=1,mode='max')
model.fit_generator(datagen.flow(x_train_zoom,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_zoom_gen,epochs=100, callbacks=[early_stopping])
_, acc, mac_f1, wei_f1 = model.evaluate(x_test_zoom,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))

# 주의할 점이 있다.
# 만약, 데이터를 생성해서 모델을 평가하고 싶으면 evaluate_generator를 사용하면 된다.
# 물론, test데이터도 train과 valid처럼 data를 generating 해야함.

#2) size= 48
# MLP model
inputs = tf.keras.Input(shape=(48,48,3))
x=tf.keras.layers.Flatten()(inputs)
x=tf.keras.layers.Dense(units=128,activation = 'relu',name='d1')(x)
x=tf.keras.layers.Dropout(0.3)(x)
x=tf.keras.layers.Dense(units=512,activation = 'relu',name='d2')(x)
x=tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(units=7,activation = tf.nn.softmax,name='d3')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',metrics=['accuracy',macro_f1score,weighted_f1score])
early_stopping = EarlyStopping(monitor='val_macro_f1score', patience=3, verbose=1,mode='max')
model.fit_generator(datagen.flow(x_train,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_gen,epochs=100, callbacks=[early_stopping])
_, acc, mac_f1, wei_f1 = model.evaluate(x_test,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))


###################### 2. model - simple CNN ######################

#1) size=64

classes = y_test.shape[1]
size=64

input_shape = (size,size,3)
batch_size = 128
kernel_size = (3,3)
filters = 64
dropout = 0.3

# CNN model
cnn_model = tf.keras.models.Sequential()
cnn_model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                     activation='relu', input_shape=input_shape, strides = (1,1) , name='Conv2D_layer1'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling1_2D'))
cnn_model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                     activation='relu', strides = (1,1) , name='Conv2D_layer2'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling2_2D'))
cnn_model.add(tf.keras.layers.Flatten(name='Flatten'))
cnn_model.add(tf.keras.layers.Dropout(dropout))
cnn_model.add(tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer'))
cnn_model.add(tf.keras.layers.Dense(classes, activation='softmax', name='Output_layer'))

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy',macro_f1score,weighted_f1score])

early_stopping = EarlyStopping(monitor='val_macro_f1score', patience=3, verbose=1,mode='max')

cnn_model.fit_generator(datagen.flow(x_train_zoom,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_zoom_gen,epochs=100, callbacks=[early_stopping])

_, acc, mac_f1, wei_f1 = cnn_model.evaluate(x_test_zoom,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))


#2) size = 48

epochs = 100
classes = y_test.shape[1]
size=48

input_shape = (size,size,3)
batch_size = 128
kernel_size = (3,3)
filters = 64
dropout = 0.3

# CNN model
cnn_model2 = tf.keras.models.Sequential()
cnn_model2.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                     activation='relu', input_shape=input_shape, strides = (1,1) , name='Conv2D_layer1'))
cnn_model2.add(tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling1_2D'))
cnn_model2.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                     activation='relu', strides = (1,1) , name='Conv2D_layer2'))
cnn_model2.add(tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling2_2D'))
cnn_model2.add(tf.keras.layers.Flatten(name='Flatten'))
cnn_model2.add(tf.keras.layers.Dropout(dropout))
cnn_model2.add(tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer'))
cnn_model2.add(tf.keras.layers.Dense(classes, activation='softmax', name='Output_layer'))

cnn_model2.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy',macro_f1score,weighted_f1score])


early_stopping = EarlyStopping(monitor='val_macro_f1score', patience=3, verbose=1,mode='max')

cnn_model2.fit_generator(datagen.flow(x_train,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_gen,epochs=100, callbacks=[early_stopping])
_, acc, mac_f1, wei_f1 = cnn_model2.evaluate(x_test,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))

