#pip install tensorflow==2.1.0

import sys
# 모듈로 받을 경로 확인
sys.path
# 내 노트북이 아닌, 전산실 컴퓨터의 colab에서 돌렸으므로, 다시돌리려면 경로 수정할것!
sys.path.append("/content/drive/My Drive/Colab Notebooks/project")
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation , Flatten
from tensorflow.keras.layers import AveragePooling2D, Input, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping , LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os



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


# Support Functions
# learning rate는 적당한 epoch를 돌리고나면, 급격히 줄임으로써 좀더 세밀한 지역의 최적화를 이어나간다.
# 고차원 loss function의 모습은 fractal처럼 맞물려있기 때문.
# 이를 learning decay라 부름.
def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

###### naive , advanced, more advanced function들에 모두 들어간다. ######


# 편의상 아래와 같은 resnet 묶음의 이름을 resnet_layer이라 하자
def resnet_layer(inputs, num_filters=16,kernel_size=(3,3),strides=(1,1),activation=None,kernel_regularizer=l2(1e-4)):

    conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)
    x = inputs
    x = conv(x)
    x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x) # conv-bn-activatio
    return x


###### advanced, more advanced function들에 모두 들어간다. ######
######  more advanced function 는 for문을 이용해서 더 줄였다. ######




# 내 데이터에 맞는 resnet18 코드를 만들기 앞서, 코드가 너무 길어 비효율적이므로 적당한 함수를 만든다.
# block_v1 은 resnet18과 resnet34에 특화된 block이고, block_v2는 resnet50, resnet101, resnet152에 특화된 block이다.
# 이 두개의 block은, 나중에 하나의 모형에서 같이 고려될 것이다.

def block_v1(input_shape, num_filters, identi, half=True, kernel_size= (3,3)):
    # identi : resnet의 존재 이유인, 더하게 될 자기자신값이다.
    if half:
        layer1 = resnet_layer(inputs=input_shape, num_filters=num_filters, strides=(2, 2), kernel_size=kernel_size,
                              activation='relu')
        layer2 = resnet_layer(inputs=layer1, num_filters=num_filters, strides=(1, 1), kernel_size=kernel_size,
                              activation=None)
    else:
        layer1 = resnet_layer(inputs=input_shape, num_filters=num_filters, strides=(1, 1), kernel_size=kernel_size,
                              activation='relu')
        layer2 = resnet_layer(inputs=layer1, num_filters=num_filters, strides=(1, 1), kernel_size=kernel_size,
                              activation=None)
    res_2_1 = add([identi, layer2])  # block 1
    output = Activation("relu")(res_2_1)
    return output


def block_v2(input_shape, num_filters, identi, half=True):
    # identi : resnet의 존재 이유인, 더하게 될 자기자신값이다.
    if half:
        layer1 = resnet_layer(inputs=input_shape, num_filters=num_filters, strides=(2, 2), kernel_size=(1, 1),
                              activation='relu')
        layer2 = resnet_layer(inputs=layer1, num_filters=num_filters, strides=(1, 1), kernel_size=(3, 3),
                              activation='relu')
        layer3 = resnet_layer(inputs=layer2, num_filters= 4*num_filters, strides=(1, 1), kernel_size=(1, 1),
                              activation=None)
    else:
        layer1 = resnet_layer(inputs=input_shape, num_filters= num_filters, strides=(1, 1), kernel_size=(1, 1),
                              activation='relu')
        layer2 = resnet_layer(inputs=layer1, num_filters= num_filters, strides=(1, 1), kernel_size=(3, 3),
                              activation='relu')
        layer3 = resnet_layer(inputs=layer2, num_filters= 4*num_filters, strides=(1, 1), kernel_size=(1, 1),
                              activation=None)
    res_2_1 = add([identi, layer3])  # block 1
    output = Activation("relu")(res_2_1)
    return output





# 1-1). Original ResNet18
def resnet18(input_shape=(224,224,3), classes=1000):

    inputs = Input(shape=input_shape)

    # conv1
    layer1 = resnet_layer(inputs=inputs,num_filters=64, strides=(2,2), kernel_size=(7,7),activation='relu')
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(layer1)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1/3x3_s2')(pool1_helper)

    # 변수, layer_a_b_c 에서    a : conv number , b : block number , c : layer number 이다.
    # 변수, res_a_b 에서        a : conv number , b : block number ( = res number) 이다.
    # 변수, acti_a_b 에서       a : conv number , b : number between blocks 이다.
    # 이 사실을 기억하고, 아래의 변수들을 해석한다.

    # conv2_x
    layer_2_1_1 = resnet_layer(inputs=pool_1, num_filters=64, strides=(1,1), kernel_size=(3,3), activation='relu')
    layer_2_1_2 = resnet_layer(inputs=layer_2_1_1, num_filters=64, strides=(1,1), kernel_size=(3,3), activation=None)
    res_2_1 = add([pool_1, layer_2_1_2]) # block 1
    acti_2_1 = Activation("relu")(res_2_1)
    layer_2_2_1 = resnet_layer(inputs=acti_2_1, num_filters=64, strides=(1,1), kernel_size=(3,3), activation='relu')
    layer_2_2_2 = resnet_layer(inputs=layer_2_2_1, num_filters=64, strides=(1,1), kernel_size=(3,3), activation=None)
    res_2_2 = add([acti_2_1, layer_2_2_2]) # block 2 // 그리고 이런 규칙은 아래에서도 적용된다.
    acti_2_output = Activation('relu')(res_2_2)

    # 변수 identi_3_input 은 conv3_x 묶음에 처음으로 들어올 input이다. 이 값은 변환 없이 자기 자신을 유지한 채로 간다.

    # conv3_x
    identi_3_input = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(acti_2_output) # 차원을 맞춰주기 위해 convolution 연산 추가. (그대로 가는것)
    #identi_3_input = BatchNormalization()(identi_3_input) # 모든 conv.뒤에는 bn을 적용.
    layer_3_1_1 = resnet_layer(inputs=acti_2_output, num_filters=128, strides=(2,2), kernel_size=(3,3),activation='relu') #이 아래 두층은 rensnet층
    layer_3_1_2 = resnet_layer(inputs=layer_3_1_1, num_filters=128, strides=(1,1), kernel_size=(3,3), activation=None)
    res_3_1 = add([identi_3_input, layer_3_1_2])
    acti_3_1 = Activation("relu")(res_3_1)
    layer_3_2_1 = resnet_layer(inputs=acti_3_1,num_filters=128, strides=(1,1), kernel_size=(3,3), activation='relu') #이 아래 두층은 rensnet층
    layer_3_2_2 = resnet_layer(inputs=layer_3_2_1, num_filters=128, strides=(1,1), kernel_size=(3,3), activation=None)
    res_3_2 = add([acti_3_1, layer_3_2_2])
    acti_3_output = Activation('relu')(res_3_2)

    # conv4_x
    identi_4_input = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(acti_3_output) # 차원을 맞춰주기 위해 convolution 연산 추가. (그대로 가는것)
    #identi_4_input = BatchNormalization()(identi_4_input) # 모든 conv.뒤에는 bn을 적용.
    layer_4_1_1 = resnet_layer(inputs=acti_3_output, num_filters=256, strides=(2,2), kernel_size=(3,3),activation='relu') #이 아래 두층은 rensnet층
    layer_4_1_2 = resnet_layer(inputs=layer_4_1_1, num_filters=256, strides=(1,1), kernel_size=(3,3), activation=None)
    res_4_1 = add([identi_4_input, layer_4_1_2])
    acti_4_1 = Activation("relu")(res_4_1)
    layer_4_2_1 = resnet_layer(inputs=acti_4_1,num_filters=256, strides=(1,1), kernel_size=(3,3), activation='relu') #이 아래 두층은 rensnet층
    layer_4_2_2 = resnet_layer(inputs=layer_4_2_1, num_filters=256, strides=(1,1), kernel_size=(3,3), activation=None)
    res_4_2 = add([acti_4_1, layer_4_2_2])
    acti_4_output = Activation('relu')(res_4_2)

    # conv5_x
    identi_5_input = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(acti_4_output) # 차원을 맞춰주기 위해 convolution 연산 추가. (그대로 가는것)
    #identi_5_input = BatchNormalization()(identi_5_input) # 모든 conv.뒤에는 bn을 적용.
    layer_5_1_1 = resnet_layer(inputs=acti_4_output, num_filters=512, strides=(2,2), kernel_size=(3,3),activation='relu') #이 아래 두층은 rensnet층
    layer_5_1_2 = resnet_layer(inputs=layer_5_1_1, num_filters=512, strides=(1,1), kernel_size=(3,3), activation=None)
    res_5_1 = add([identi_5_input, layer_5_1_2])
    acti_5_1 = Activation("relu")(res_5_1)
    layer_5_2_1 = resnet_layer(inputs=acti_5_1,num_filters=512, strides=(1,1), kernel_size=(3,3), activation='relu') #이 아래 두층은 rensnet층
    layer_5_2_2 = resnet_layer(inputs=layer_5_2_1, num_filters=512, strides=(1,1), kernel_size=(3,3), activation=None)
    res_5_2 = add([acti_5_1, layer_5_2_2])
    acti_5_output = Activation('relu')(res_5_2)

    # 마지막 1층
    y=AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='loss1/ave_pool')(acti_5_output)
    y = Flatten()(y)
    outputs = Dense(classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# 아래의 resnet18은, block_v1이 사용되었다.
# 함수의 이름에서 주목할 수 있듯, 이 모형은 resnet18밖에 사용하지 못함.
# 이를 해결한 모형은 2-3)에서 더 간단한 형태로 제시된다.


def resnet18(input_shape=(224, 224, 3), classes=1000):
    inputs = Input(shape=input_shape)

    # conv1
    layer1 = resnet_layer(inputs=inputs, num_filters=64, strides=(2, 2), kernel_size=(7, 7), activation='relu')
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(layer1)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')(pool1_helper)

    # 변수, block_a_b 에서    a : conv number , b : block number 이다.
    # conv2_x
    block_2_1 = block_v1(input_shape=pool_1, num_filters=64, identi=pool_1, half=False, kernel_size=(3, 3))
    block_2_2 = block_v1(input_shape=block_2_1, num_filters=64, identi=block_2_1, half=False, kernel_size=(3, 3))

    # conv3_x
    identi_3_input = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(block_2_2)  # 차원을 맞춰주기 위해 convolution 연산 추가. (그대로 가는것)
    #identi_3_input = BatchNormalization()(identi_3_input) # 모든 conv.뒤에는 bn을 적용.
    block_3_1 = block_v1(input_shape=block_2_2, num_filters=128, identi=identi_3_input, half=True, kernel_size=(3, 3))
    block_3_2 = block_v1(input_shape=block_3_1, num_filters=128, identi=block_3_1, half=False, kernel_size=(3, 3))

    # conv4_x
    identi_4_input = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(block_3_2)  # 차원을 맞춰주기 위해 convolution 연산 추가. (그대로 가는것)
    #identi_4_input = BatchNormalization()(identi_4_input) # 모든 conv.뒤에는 bn을 적용.
    block_4_1 = block_v1(input_shape=block_3_2, num_filters=256, identi=identi_4_input, half=True, kernel_size=(3, 3))
    block_4_2 = block_v1(input_shape=block_4_1, num_filters=256, identi=block_4_1, half=False, kernel_size=(3, 3))

    # conv5_x
    identi_5_input = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(block_4_2)  # 차원을 맞춰주기 위해 convolution 연산 추가. (그대로 가는것)
    #identi_5_input = BatchNormalization()(identi_5_input) # 모든 conv.뒤에는 bn을 적용.
    block_5_1 = block_v1(input_shape=block_4_2, num_filters=512, identi=identi_5_input, half=True, kernel_size=(3, 3))
    block_5_2 = block_v1(input_shape=block_5_1, num_filters=512, identi=block_5_1, half=False, kernel_size=(3, 3))

    # 마지막 1층
    y = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='loss1/ave_pool')(block_5_2)
    y = Flatten()(y)
    outputs = Dense(classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# 좀더 깔끔하게 작성됨. 앞으로의 resnet모형은 다음의 함수에서 해결한다.

def resnet(input_shape=(224, 224, 3), num_filters=64, classes=1000, num_blocks=[0, 0, 0, 0], version=None):
    # num_blocks 는 list로 받는 인자. conv3_x ~ conv5_x의 block의 수를 앞에서부터 차례로 넣어주면 된다.
    # num_filters 는 초기의 filters를 의미한다.

    if len(num_blocks) != 4:
        raise NameError("Please input the number of blocks from conv2_x to conv5_x in the 'num_blocks' variable.")

    if version == "v1":  # resnet 18, 34에 해당. (블록이 같음)
        block = block_v1
    elif version == "v2":  # resnet 50, 101, 152에 해당. (블록이 같음)
        block = block_v2
    else:
        raise NameError("Please input the string 'v1' or 'v2' in the 'version' variable. ")

    inputs = Input(shape=input_shape)

    # conv1
    layer1 = resnet_layer(inputs=inputs, num_filters=num_filters, strides=(2, 2), kernel_size=(7, 7), activation='relu')
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(layer1)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    cum_block = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')(pool1_helper)

    # cum_block 변수에 하나씩 residual block을 누적시켜보자.

    for stack in range(4):
        if stack == 0:
            # 우리가 누적시킬 변수, cum_block을 정의하고, 이 변수에 누적시켜가면서 모형을 출력한다.

            # 이 아래 for문은 conv2_x
            # 들어가기 전에, block_v2라면 input과 output의 filter size를 동일하게 해주는 작업을 한다. 덧셈이 가능해야 하므로, 반드시!
            if version == 'v2':
                cum_block = Conv2D(4 * num_filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                   kernel_initializer='he_normal')(cum_block)
                # cum_block = BatchNormalization()(cum_block) # 모든 conv.뒤에는 bn을 적용.

            for res_block in range(num_blocks[stack]):
                cum_block = block(input_shape=cum_block, num_filters=num_filters, identi=cum_block, half=False)
        else:

            # 이 아래는 conv3_x ~ conv5_x 까지.
            num_filters *= 2

            if version == 'v2':
                x = Conv2D(4 * num_filters, kernel_size=(1, 1), strides=(2, 2), padding='valid',
                           kernel_initializer='he_normal')(cum_block)  # block_v2라면, input에 해당하는 x를 덧셈이 가능하게 맞춰준다.
                # x = BatchNormalization()(x) # 모든 conv.뒤에는 bn을 적용.
            else:
                x = Conv2D(num_filters, (1, 1), strides=(2, 2), padding='valid')(
                    cum_block)  # identi_input 에 해당 // rxc가 각각 반토막남.
                # 즉, 위 x는 결국 이어질 conv 층의 input임.

            for res_block in range(num_blocks[stack]):
                if res_block == 0:
                    cum_block = block(input_shape=cum_block, num_filters=num_filters, identi=x, half=True)  # block 갱신
                else:
                    cum_block = block(input_shape=cum_block, num_filters=num_filters, identi=cum_block,
                                      half=False)  # block 갱신

    # 마지막 1층
    y = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='loss1/ave_pool')(cum_block)
    y = Flatten()(y)
    outputs = Dense(classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# resnet18
model = resnet(input_shape=(224,224,3),classes=1000,num_blocks=[2,2,2,2],version='v1')

model.summary()

# 여기는 학습의 효율을 위해. Adam으로 넘어간다.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy',macro_f1score,weighted_f1score])

early_stopping = EarlyStopping(monitor = 'val_macro_f1score',patience = 3 , verbose=1,mode='max')
lr_sc = LearningRateScheduler(lr_schedule,verbose=1)
model.fit(datagen.flow(x_train_zoom,y_train,batch_size=128), steps_per_epoch=len(x_train)/128, validation_data= xy_valid_zoom_gen, validation_steps=len(x_valid_zoom)/128, epochs=100, callbacks=[early_stopping,lr_sc])
_, acc, mac_f1, wei_f1 = model.evaluate(x_test_zoom,y_test,batch_size=128)
print("\nAccuracy: {:.4f}, Macro F1 Score: {:.4f}, Weighted F1 Score: {:.4f}".format(acc,mac_f1,wei_f1))






#graph
from tensorflow.keras.utils import plot_model

# 만약, plot_model이 작동하지 않으면, 아래의 절차를 밟아본다.

#  1. pip install pydot // pip install graphviz 설치
#  2. 그래도 작동이 안되면, 아래처럼 환경변수 추가하고 실행.
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

plot_model(model, to_file='model_shapes.png', show_shapes=True) #현재 경로에 저장

