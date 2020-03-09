import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img , img_to_array , ImageDataGenerator
from numpy import expand_dims

# data augmentation
# 1. 보통은 상하반전 대신, 좌우반전을 한다. 그래도 넣어서 나쁠건 없음.
# 2. cnn의 크기가 어느정도 커지면 회전이 들어가도 분류를 잘하기에, 회전은 잘 하지 않는다. 그래도 넣어서 나쁠건 없음.



# 0 . data import
img1=load_img("C:/Users/82104/Desktop/fer2013/all_images/00000001.jpg")
data = img_to_array(img1)
data.shape # 자동으로 rgb로 만들어줌. (채널 : 3)

plt.imshow(img1)
# plt.imshow(data/255) 와 같다.

# 1.  horizontal and vertical shift augmentation
# 1) horizontal
samples = data[np.newaxis,:]
# samples = np.expand_dims(data,0) 과 같음.

datagen = ImageDataGenerator(width_shift_range=[-10,10])
# 움직일 폭을 결정. -10 또는 10 pixel 만 정확하게 움직임.
# 만약 0과 1사이의 scalar 값을 주게되면, pixel의 해당 비율로 인식하여
# 그 값 내에서 좌우 이동을 시킨다.

it = datagen.flow(samples,batch_size=1)
# it은 iter의 약자
# datagen.flow(x_train,y_train,batch_size = 32)
# batch_size 만큼 데이터 생성
fig = plt.figure(figsize=(11,11)) # 9개짜리 figure받을 공간 생성

for i in range(9) :
    plt.subplot(3,3,i+1)
    batch = it.next()
    # next는 자체의 함수 혹은 매서드. datagen.flow에서는 method로 갖는다.
    # 자체로는 못쓰고, 보통 iter함수와 같이 쓴다.
    # 즉, iterator로 만들어주고, 거기서 next를 쓰는것.

    image = batch[0].astype('uint8')
    # batch에는 1개 값만 저장되고, 그 값의 차원은 (1,48,48,3) 임.
    # batch[0]은 (48,48,3)의 차원을 출력.
    plt.imshow(image)

# 2) vertical
samples = data[np.newaxis,:]
datagen = ImageDataGenerator(height_shift_range=0.5)
# height_shift_range 는 width_shift_range 와 같다.

it = datagen.flow(samples,batch_size=1)
fig = plt.figure(figsize=(11,11))

for i in range(9):
    plt.subplot(330+1+i)
    # 331 ~ 339 로, (3,3,1)~(3,3,9) 를 세자리 정수로도 받을 수 있음.

    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)


# 2.  horizontal and vertical flip augmentation

samples = expand_dims(data,0)
datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True)
it = datagen.flow(samples,batch_size=1)
fig = plt.figure(figsize=(11,11))

for i in range(9):
    plt.subplot(330+1+i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

# 3. random rotation augmentation

samples = expand_dims(data,0)
datagen = ImageDataGenerator(rotation_range=90)
# 좌,우로 최대 90도까지 회전할수 있다는 뜻.
it = datagen.flow(samples,batch_size=1)

fig = plt.figure(figsize=(11,11))

for i in range(9):
    plt.subplot(330+1+i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

# 4. random brightness augmentation

samples = expand_dims(data,0)
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# 0이 제일 어두운 값이며, 1이 제일 밝은 값.
# 이 안에서 밝기를 조절하게 됨.

it = datagen.flow(samples,batch_size=1)

fig = plt.figure(figsize=(11,11))
for i in range(9):
    plt.subplot(330+1+i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

# 5. random zoom augmentation

samples = expand_dims(data,0)
datagen = ImageDataGenerator(zoom_range=[0.5,1])
# 소수 혹은 정수를 다 받음.
# 값이 클수록 실제 사진은 zoom됨. 즉 작아지는거임.
# 1의 값이 기본.

it = datagen.flow(samples,batch_size=1)


fig = plt.figure(figsize=(11,11))
for i in range(9):
    plt.subplot(330+1+i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()


# 6. all together

samples = expand_dims(data,0)
datagen = ImageDataGenerator(
    zoom_range=[0.5,1.0],
    brightness_range=[0.5,1.0],
    rotation_range=30,
    horizontal_flip = True,
    vertical_flip = True,
    height_shift_range = 0.1,
    width_shift_range = 0.1)

it = datagen.flow(samples,batch_size=1)
fig = plt.figure(figsize=(13,13))

for i in range(12):
    plt.subplot(4,3,1+i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)

plt.show()


# 7. example

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="constant",
    cval=0)
datagen.fit(x_train)

# simple cnn
inputs = tf.keras.Input(shape=(32,32,3))
x=tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3),
                                     activation='relu', input_shape=(32,32,3), strides = (1,1) , name='Conv2D_layer1')(inputs)
x=(tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling1_2D'))(x)
x=tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3),
                                     activation='relu', strides = (1,1) , name='Conv2D_layer2')(x)
x= tf.keras.layers.MaxPooling2D((2, 2), name='Maxpooling2_2D')(x)
x = tf.keras.layers.Flatten(name='Flatten')(x)
x=tf.keras.layers.Dropout(0.3)(x)
x=tf.keras.layers.Dense(32, activation='relu', name='Hidden_layer')(x)
outputs =tf.keras.layers.Dense(10, activation='softmax', name='Output_layer')(x)




## data argumentation
model=tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),
                    steps_per_epoch=len(x_train)/32,epochs=1)
_, acc = model.evaluate(x_test,y_test,batch_size=32)
print("\nAccuracy: {:.4f}, F1 Score: {:.4f}".format(acc,1))

## normal data
model2=tf.keras.Model(inputs=inputs,outputs=outputs)
model2.summary()
model2.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model2.fit(x_train,y_train, batch_size=32,epochs=1)

_, acc = model2.evaluate(x_test,y_test,batch_size=32)
print("\nAccuracy: {:.4f}, F1 Score: {:.4f}".format(acc,1))