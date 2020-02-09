import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

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



###################### 1. model - MLP ######################

#data import
x_train = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\X_train.csv",
                      header=0,index_col=0)
x_valid = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\X_private_test.csv",
                      header=0,index_col=0)
x_test = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\X_public_test.csv",
                      header=0,index_col=0)
y_train = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\y_train.csv",
                      header=0,index_col=0)
y_valid = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\y_private_test.csv",
                      header=0,index_col=0)
y_test = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\y_public_test.csv",
                      header=0,index_col=0)

# data handling
x_train = np.array(x_train).reshape([-1,48,48]) / 255
x_valid = np.array(x_valid).reshape([-1,48,48]) / 255
x_test = np.array(x_test).reshape([-1,48,48]) / 255
y_train = np.array(y_train).reshape([-1,])
y_valid = np.array(y_valid).reshape([-1,])
y_test = np.array(y_test).reshape([-1,])

epochs = 3
classes = len(np.unique(y_test))

# MLP model
inputs = tf.keras.Input(shape=(48,48))
x=tf.keras.layers.Flatten()(inputs)
x=tf.keras.layers.Dense(units=128,activation = 'relu',name='d1')(x)
x=tf.keras.layers.Dropout(0.3)(x)
x=tf.keras.layers.Dense(units=512,activation = 'relu',name='d2')(x)
x=tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(units=classes,activation = tf.nn.softmax,name='d3')(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',metrics=['accuracy',f1score])
model.fit(x_train,y_train,batch_size=128,epochs=epochs)

_, acc, f1 = model.evaluate(x_test,y_test,batch_size=batch_size)
print("\nAccuracy: {:.4f}, F1 Score: {:.4f}".format(acc,f1))
##################################################################


###################### 2. model - CNN Basic ######################

#data import
x_train = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\X_train.csv",
                      header=0,index_col=0)
x_valid = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\X_private_test.csv",
                      header=0,index_col=0)
x_test = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\X_public_test.csv",
                      header=0,index_col=0)
y_train = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\y_train.csv",
                      header=0,index_col=0)
y_valid = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\y_private_test.csv",
                      header=0,index_col=0)
y_test = pd.read_csv("C:\\Users\\82104\\Desktop\\fer2013\\mydata\\y_public_test.csv",
                      header=0,index_col=0)

#data handling
x_train = np.array(x_train).reshape([-1,48,48,1]) / 255
x_valid = np.array(x_valid).reshape([-1,48,48,1]) / 255
x_test = np.array(x_test).reshape([-1,48,48,1]) / 255
y_train = np.array(y_train).reshape([-1,])
y_valid = np.array(y_valid).reshape([-1,])
y_test = np.array(y_test).reshape([-1,])

epochs = 3
image_size=48
classes = len(np.unique(y_test))

input_shape = (image_size,image_size,1)
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
                                     activation='relu', input_shape=input_shape, strides = (1,1) , name='Conv2D_layer2'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling2_2D'))
cnn_model.add(tf.keras.layers.Flatten(name='Flatten'))
cnn_model.add(tf.keras.layers.Dropout(dropout))
cnn_model.add(tf.keras.layers.Dense(64, activation='relu', name='Hidden_layer'))
cnn_model.add(tf.keras.layers.Dense(classes, activation='softmax', name='Output_layer'))

cnn_model.summary()

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy',f1score])
cnn_model.fit(x_train,y_train, batch_size=batch_size,epochs=epochs)

_, acc, f1 = model.evaluate(x_test,y_test,batch_size=batch_size)
print("\nAccuracy: {:.4f}, F1 Score: {:.4f}".format(acc,f1))
##################################################################