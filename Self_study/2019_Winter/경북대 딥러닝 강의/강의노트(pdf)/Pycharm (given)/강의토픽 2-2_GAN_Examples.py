"""
2-2-1 MNIST 이미지 생성을 위한 기본 GAN 구현

참고교재: 텐서플로로 배우는 딥러닝
"""
# 필요한 라이브러리를 불러들임
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 읽어들임
from tensorflow.keras.datasets import mnist
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

xtrain = xtrain.reshape(-1,784)
xtest = xtest.reshape(-1,784)

from tensorflow.keras.utils import to_categorical
ytrain = to_categorical(ytrain,10)
ytest = to_categorical(ytest,10)

# 학습 관련 매개변수 설정
n_noise = 100
n_h1 = 150
n_h2 = 300
batch_size = 256
n_epoch = 60

# 생성자 정의 : 진짜 데이터와 유사한 가짜 데이터 생성
def generator(g_shape):
    input_x = tf.keras.layers.Input(g_shape)
    net = tf.keras.layers.Dense(n_h1,activation='relu',bias_initializer='zeros',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))(input_x) # hidden1
    net = tf.keras.layers.Dense(n_h2,activation='relu',bias_initializer='zeros',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))(net) # hidden2
    net = tf.keras.layers.Dense(784,activation='tanh',bias_initializer='zeros',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))(net) 
    G = tf.keras.Model(input_x,net)
    return G

# 판별자 정의: 진짜 이미지와 가짜 이미지 분류
def discriminator(d_shape):
    input_x = tf.keras.layers.Input(d_shape)
    net = tf.keras.layers.Dense(n_h1,activation='relu',bias_initializer='zeros',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))(input_x) # hidden 1
    net = tf.keras.layers.Dropout(0.3)(net) # hidden 1 dropout
    net = tf.keras.layers.Dense(n_h2,bias_initializer='zeros',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))(net) # hidden 2
    net = tf.keras.layers.Dense(0.3)(net) # hidden 2 dropout
    net = tf.keras.layers.Dense(batch_size,activation='sigmoid',bias_initializer='zeros',
                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))(net)
    D = tf.keras.Model(input_x,net)
    return D

g_shape = (n_noise,)
G = generator(g_shape)
G.summary()

d_shape = (784,)
D = discriminator(d_shape)
D.summary()

# 손실함수 및 최적화 방법 정의
def d_loss(y_data,y_fake):
    return -(tf.math.log(y_data)+tf.math.log(1-y_fake))

def g_loss(y_fake):
    return -tf.math.log(y_fake)

def train():
    optimizer = tf.keras.optimizers.Adam(0.0001)
    
    @tf.function
    def train_step(x,z_noise):
        with tf.GradientTape(persistent=True) as tape:       
            fake_data = G(z_noise)
            d_fake_data = D(fake_data)
            d_real_data = D(x)
            d_loss_value = d_loss(d_real_data, d_fake_data)
            g_loss_value = g_loss(d_fake_data)
        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        del tape
        optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
        optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))
        return fake_data, g_loss_value, d_loss_value
    
    for epoch in range(n_epoch):
        n_batch = int(len(xtrain)/batch_size)
        
        for i in range(n_batch):
            z_noise = tf.random.uniform((batch_size,n_noise),-1,1)
            batch_x = xtrain[i*batch_size:(i+1)*batch_size]
            x_value = 2*batch_x.astype(np.float32)-1
            fake_data,g_loss_value,d_loss_value = train_step(x_value,z_noise)
            
        if epoch % 5 == 0:
            print("G loss: ", g_loss_value, " D loss: ", d_loss_value, " epoch: ", epoch)
        
    z_sample = tf.random.uniform((batch_size,n_noise),-1,1)
    out_img = G(z_sample)
    imgs = 0.5*(out_img+1)
        
    for k in range(25):
        plt.subplot(5,5,k+1)
        image = np.reshape(imgs[k],(28,28))
        plt.imshow(image,cmap='gray')
        
train()


"""
2-2-2 인물 이미지 생성을 위한 기본 LSGAN 구현

참고교재: 텐서플로로 배우는 딥러닝

이미지 데이터는 크기가 45*40인 컬러 인물 이미지 20장이다. GAN을 실습하는 데이터로는 
부족한 면이 있지만, 프로그램에 대한 이해를 돕기 위해서는 짧은 시간에 결과를 볼 수 
있는 것이 좋을 것으로 생각되어 사용하였다. 더 많은 학습을 위한 GAN 실습에 좋은 
이미지 데이터를 인터넷에서 구하는 것은 어렵지 않을 것이다.

생성자는 입력층, 2개의 은닉층 및 출력층으로 구성된 신경망이다. 인물 이미지는 
45*40*3 = 5400 픽셀의 이미지이므로 생성자의 출력층은 5,400개의 출력노드를 가진다. 
시그모이드 함수와 비교했을 때 쌍곡탄젠트 함수가 기울기 소멸 문제로 인한 어려움을 
덜 겪으므로 출력노드에 적용되는 활성함수로 시그모이드 함수 대신 쌍곡탄젠트 함수를 사용한다. 
실제 인물 이미지와 생성자에 의해 만들어진 가짜 이미지를 분류하는 판별자는 입력층, 2개의
은닉층 및 출력층으로 구성된 신경망이다. 출력노드에 적용되는 활성함수로 시그모이드 함
수를 사용하고 각 은닉층에는 드롭아웃을 적용한다. 생성자의 입력벡터, 즉 잡음은 균등분
포 U(-1,1)로부터 추출된 값으로 이루어진 100차원의 벡터이다.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as Im
import glob
import os

tf.random.set_seed(1)

# data
os.chdir('G:/Data/')
files=glob.glob('./face20/*.png')
img=[]
for file in files:
    temp=Im.open(file)
    temp=np.array(temp)
    temp=temp[:,:,0:3]/255.
    img.append(temp)
    
# 리스트를 array
x=np.asarray(img).astype(np.int)
n_cell=np.prod(x.shape[1:4]).astype(np.int)
# 이미지를 벡터
x_vec=np.reshape(x,[len(img),n_cell])

sample_size=x_vec.shape[0] # 20
input_dim=x_vec.shape[1] # 5400

# 매개변수
learning_rate=0.001
batch_size=20
z_size=100
nepochs=5000
g_hidden_size=128
d_hidden_size=128

def generator():
    input_x=tf.keras.Input([z_size,])
    net=tf.keras.layers.Dense(g_hidden_size,activation='relu',
                              bias_initializer='zeros',kernel_initializer='glorot_uniform')(input_x)
    net=tf.keras.layers.Dense(input_dim,kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',activation='tanh')(net)
    G=tf.keras.Model(input_x,net)
    return G

def discriminator():
    input_x=tf.keras.Input([input_dim,])
    net=tf.keras.layers.Dense(d_hidden_size,activation='relu',
                              bias_initializer='normal',kernel_initializer='glorot_uniform')(input_x)
    net=tf.keras.layers.Dropout(0.2)(net)
    net=tf.keras.layers.Dense(1,bias_initializer='normal',kernel_initializer='glorot_uniform')(net)
    net=tf.keras.layers.Dropout(0.2)(net)
    net=tf.keras.activations.sigmoid(net)
    D=tf.keras.Model(input_x,net)
    return D

G=generator()
D=discriminator()

def d_loss(d_real,d_fake):
    return 0.5*(tf.reduce_mean(d_real-1)**2)+0.5*tf.reduce_mean(d_fake**2)

def g_loss(d_fake):
    return 0.5*tf.reduce_mean((d_fake-1)**2)

def train():
    global d_loss_value
    global g_loss_value
    d_solver=tf.keras.optimizers.Adam(learning_rate)
    g_solver=tf.keras.optimizers.Adam(learning_rate)
    
    @tf.function
    def gradient(x,z):

        with tf.GradientTape(persistent=True) as tape:
            fake_data=G(z)
            d_fake_data=D(fake_data)
            d_real_data=D(x)
            d_loss_value = d_loss(d_real_data, d_fake_data)
            g_loss_value = g_loss(d_fake_data)
        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        d_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
        g_solver.apply_gradients(zip(g_gradients, G.trainable_variables))
        return d_loss_value, g_loss_value
    
    losses=[]
    for epoch in range(nepochs):
        n_batch=int(sample_size/batch_size)
        
        for ii in range(n_batch):
            if ii != n_batch:
                batch_images=x_vec[ii*batch_size:(ii+1)*batch_size]
            else:
                batch_images=x_vec[(ii+1)*batch_size:]
            
            batch_images=batch_images*2-1
            batch_z=tf.random.uniform((batch_size,z_size),-1,1)
            
            d_loss_value,g_loss_value=gradient(batch_images,batch_z)
            d_loss_value=np.array(d_loss_value)
            g_loss_value=np.array(g_loss_value)
            
            loss=d_loss_value+g_loss_value
        
        # print('epoch:{0},discrimniator loss: {1:7.4f},generator loss: {2:7.4f}'.format(epoch+1,d_loss_value,g_loss_value))
        losses.append((d_loss_value,g_loss_value))
       
        if (epoch+1)%100 ==0:
            tf.random.set_seed(0)
            sample_z=tf.random.uniform((20,z_size),-1,1)
            gen_samples=G(sample_z)
            f,axes=plt.subplots(figsize=(7,7),nrows=5,ncols=4,sharey=True,sharex=True)
            f.suptitle(epoch+1)
            for ii in range(20):
                plt.subplot(5,4,ii+1)
                gs=tf.reshape(gen_samples[ii],(45,40,3))
                gs=(gs-np.min(gs))/(np.max(gs)-np.min(gs))
                plt.imshow(gs)
    return losses

losses=train()

fig,ax=plt.subplots(figsize=(7,7))
losses=np.array(losses)
plt.plot(losses.T[0],'r-',label='discrimniator')
plt.plot(losses.T[1],'-b',label='generator')

f,axes=plt.subplots(figsize=(7,7),nrows=5,ncols=4,sharey=True,sharex=True)
for ii in range(20):
    plt.subplot(5,4,ii+1)
    gs=x_vec[ii].reshape(45,40,3)
    gs=(gs-np.min(gs))/(np.max(gs)-np.min(gs))
    plt.imshow(gs)    