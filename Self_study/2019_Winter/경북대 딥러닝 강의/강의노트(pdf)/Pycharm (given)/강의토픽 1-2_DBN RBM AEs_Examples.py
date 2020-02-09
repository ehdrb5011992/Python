"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)
예제 2-1. MNIST 데이터에 대한 이진 입력 RBM

입력값이 0과 1 사이의 실숫값인 28*28 MNIST 데이터에 이진 입력 RBM을 적용할 
수 있다는 것을 보이고 차원 축약을 통해 각 이미지의 숨겨진 구조를 파악한다.
"""

# 필요한 라이브러리를 불러들임
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# 학습 관련 매개변수 설정 
n_input = 784
n_hidden = 500
batch_size = 256
lr = tf.constant(0.001,tf.float32)
display_step = 2

# MNIST 파일 읽어들임
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# print(y_train[mnist_train[1] == 8])
# y_test[mnist_test[1] == 8]

# 가중치 및 편향을 정의
w = tf.Variable(tf.keras.backend.random_normal([n_input,n_hidden],0.01),name="w")
b_h = tf.Variable(tf.zeros([1,n_hidden],tf.float32,name="b_h"))
b_i = tf.Variable(tf.zeros([1,n_input],tf.float32,name='b_i'))


# 확률을 이산 상태, 즉 0과 1로 변환함
def binary(probs):
    return tf.floor(probs+tf.keras.backend.random_uniform(tf.shape(probs),0,1))

# Gibbs 표본 추출 단계
def cd_step(x_k):
    h_k = binary(tf.sigmoid(tf.matmul(x_k,w)+b_h))
    x_k = binary(tf.sigmoid(tf.matmul(h_k,tf.transpose(w))+b_i))
    return x_k

# 표본 추출 단계 실행
def cd_gibbs(k,x_k):
    for i in range(k):
        x_out = cd_step(x_k)
    # k 반복 후에 깁스 표본을 반환함
    return x_out

# CD-2 알고리즘
def cd(x):
    x_s = cd_gibbs(2,x)
    act_h_s = tf.sigmoid(tf.matmul(x_s,w)+b_h)
    act_h = tf.sigmoid(tf.matmul(x,w)+b_h)
    # 입력값이 주어질 때 은닉노드값 act_h
    # 은닉노드 값이 주어질 때 입력값 추출
    _x = binary(tf.sigmoid(tf.matmul(act_h,tf.transpose(w))+b_i))
    w_add = tf.multiply(lr/batch_size,tf.subtract(tf.matmul(tf.transpose(x),act_h),
                                             tf.matmul(tf.transpose(x_s),act_h_s)))
    bi_add = tf.multiply(lr/batch_size,tf.reduce_sum(tf.subtract(x,x_s),0,True))
    bh_add = tf.multiply(lr/batch_size,tf.reduce_sum(tf.subtract(act_h,act_h_s),0,True))
    updt = [w.assign_add(w_add),b_i.assign_add(bi_add),b_h.assign_add(bh_add)] 
    return updt, _x

# 훈련용 이미지 데이터 사용하여 학습 시작 
total_batch = int(len(x_train)/batch_size)
num_epochs = 50 # 100까진 안됨 54정도에서 멈춤 

for epoch in range(num_epochs):
    for i in range(total_batch):
        batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
        # 가중치 업데이트 
        batch_xs = (batch_xs >0)*1
        updt,_x = cd(batch_xs.astype('float32'))
        
    if epoch % display_step == 0:
        print("Epoch:",'%04d'%(epoch+1))
        
print("RBM training Completed!")

# 20개의 검정용 이미지에 대해 은닉노드의 값을 계산 
xt = (x_test[:20]>0)*1
out = tf.sigmoid(tf.matmul(xt.astype('float32'),w)+b_h)
label = y_test[:20]


# 20개의 실제 검정용 이미지 그리기
plt.figure(1)
for k in range(20):
    plt.subplot(4,5,k+1)
    image = (x_test[k]>0)*1
    image = np.reshape(image,(28,28))
    plt.imshow(image,cmap='gray')
    
# 20개의 생성된 검정용 이미지 그리기
plt.figure(2)
for k in range(20):
    plt.subplot(4,5,k+1)
    image = binary(tf.sigmoid(tf.matmul(np.reshape(out[k],(-1,n_hidden)),tf.transpose(w))+b_i))
    image = np.reshape(image,(28,28))
    plt.imshow(image,cmap='gray')
    print(np.argmax(label[k]))
    
w_out = w
    

"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)
예제 2-2. 분류용-DBN

훈련용 60,000개와 검정용 10,000개로 이루어진 MNIST 데이터에 첫 번째 은닉층의 은닉
노드 500개, 두 번째 은닉층의 은닉노드 256개와 10개의 출력노드를 가진 분류용-DBN을
적용하여 검정용 이미지에 대한 정확도를 살펴본다.
"""

# 필요한 라이브러리를 불러들임
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import time

# MNIST 파일을 읽어들임
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

# 학습 관련 매개변수 설정
n_input = 784
n_hidden1 = 500
n_hidden2 = 256
display_step = 2
n_epoch = 10
batch_size = 256
lr_rbm = tf.constant(0.001,tf.float32)
lr_class = tf.constant(0.01,tf.float32)
n_class = 10
n_iter = 200

# 첫 번째 은닉층 관련 가중치 및 편향을 정의함
w1 = tf.Variable(tf.random.normal([n_input,n_hidden1],0.01),name="w1")
b1_h = tf.Variable(tf.zeros([1,n_hidden1],tf.float32,name="b1_h"))
b1_i = tf.Variable(tf.zeros([1,n_input],tf.float32,name='b1_i'))

# 두 번째 은닉층 관련 가중치 및 편향을 정의함
w2 = tf.Variable(tf.random.normal([n_hidden1,n_hidden2],0.01),name="w2")
b2_h = tf.Variable(tf.zeros([1,n_hidden2],tf.float32,name="b2_h"))
b2_i = tf.Variable(tf.zeros([1,n_hidden1],tf.float32,name='b2_i'))

# 라벨층 관련 가중치 및 편향을 정의함
w_c = tf.Variable(tf.random.normal([n_hidden2,n_class],0.01),name='w_c')
b_c = tf.Variable(tf.zeros([1,n_class],tf.float32,name='b_c'))

# 확률을 이산 상태, 즉 0과 1로 변환함
def binary(probs):
    return tf.floor(probs+tf.random.uniform(tf.shape(probs),0,1))

# Gibbs 표본 추출 단계
def cd_step(x_k,w,b_h,b_i):
    h_k = binary(tf.sigmoid(tf.matmul(x_k,w)+b_h))
    x_k = binary(tf.sigmoid(tf.matmul(h_k,tf.transpose(w))+b_i))
    return x_k

# 표본추출 단계 실행
def cd_gibbs(k,x_k,w,b_h,b_i):
    for i in range(k):
        x_out = cd_step(x_k,w,b_h,b_i)
    # k 반복 후에 깁스 표본을 반환
    return x_out

# CD-2 알고리즘
def cd(x):
    x_s = cd_gibbs(2,x,w1,b1_h,b1_i)
    act_h1_s = binary(tf.sigmoid(tf.matmul(x_s,w1)+b1_h))
    h1_s = cd_gibbs(2,act_h1_s,w2,b2_h,b2_i)
    act_h2_s = binary(tf.sigmoid(tf.matmul(h1_s,w2)+b2_h))
    
# 입력값이 주어질 때 은닉노드값 act_h    
    act_h1 = tf.sigmoid(tf.matmul(x,w1)+b1_h)
    act_h2 = tf.sigmoid(tf.matmul(act_h1_s,w2)+b2_h)
    
# 경사 하강법을 이용한 가중치 및 편향 업데이트
    size_batch = tf.cast(tf.shape(x)[0],tf.float32)
    w1_add = tf.multiply(lr_rbm/batch_size,tf.subtract(tf.matmul(tf.transpose(x),act_h1),
                                             tf.matmul(tf.transpose(x_s),act_h1_s)))
    b1_i_add = tf.multiply(lr_rbm/batch_size,tf.reduce_sum(tf.subtract(x,x_s),0,True))
    b1_h_add = tf.multiply(lr_rbm/batch_size,tf.reduce_sum(tf.subtract(act_h1,act_h1_s),0,True))
    
    w2_add = tf.multiply(lr_rbm/batch_size,tf.subtract(tf.matmul(tf.transpose(act_h1_s),act_h2),
                                             tf.matmul(tf.transpose(h1_s),act_h2_s)))
    b2_i_add = tf.multiply(lr_rbm/batch_size,tf.reduce_sum(tf.subtract(act_h1_s,h1_s),0,True))
    b2_h_add = tf.multiply(lr_rbm/batch_size,tf.reduce_sum(tf.subtract(act_h2,act_h2_s),0,True))
    
    updt = [w1.assign_add(w1_add), b1_i.assign_add(b1_i_add), b1_h.assign_add(b1_h_add),
            w2.assign_add(w2_add), b2_i.assign_add(b2_i_add), b2_h.assign_add(b2_h_add)] 
    return act_h2, updt

# 클래스 변수 y를 one-hot 벡터로 만듬 
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

# 소프트맥스 층의 w_c, b_c 학습을 위한 손실함수 계산
def loss(x, y, weights, bias):
    logs = tf.matmul(x,weights) + bias
    logits = tf.nn.softmax(logs, 1)
    entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(logits), axis=[1]))
    return entropy_loss   # cross entropy loss 계산

# 소프트맥스 층의 w_c, b_c 학습을 위한 손실함수 계산
def gradient(x, y, weights, bias):
    with tf.GradientTape() as tape:
        loss_value = loss(x, y, weights, bias)
    return tape.gradient(loss_value, [weights, bias])# direction and value of the gradient of our weight and bias


#--------------------------------------------------------------------------------------
# RBM을 쌓아 올려가며 DBN을 Pre-training 한 후에 소프트맥스 충의 w_c, b_c를 학습하는 과정
#--------------------------------------------------------------------------------------
total_batch = int(len(x_train)/batch_size)
num_epochs = 20# 100까진 안됨 54정도에서 멈춤 

optim = tf.keras.optimizers.Adam()

for epoch in range(num_epochs):
    for i in range(total_batch):
        batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
        # 2개 RBM의 가중치 업데이트 
        batch_xs = (batch_xs >0)*1
        act_h2, updt = cd(batch_xs.astype('float32'))
        
        # DBN 위에 추가된 소프트맥스 층의 w_c, b_c 업데이트
        delw_c, delb_c = gradient(act_h2, batch_ys, w_c, b_c)  
        change_w_c = delw_c * lr_class   # adjustment amount for weight
        change_b_c = delb_c * lr_class   # adjustment amount for bias
        w_c.assign_sub(change_w_c)           # subract from w_c
        b_c.assign_sub(change_b_c)           # subract from b_c
        
        loss_train = loss(act_h2, batch_ys, w_c, b_c)  
        
        logits = tf.nn.softmax(tf.matmul(act_h2,w_c) + b_c, 1)
        correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(batch_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        #print("accuracy at step: ",i," is: ",accuracy,loss)
        
    if epoch % display_step == 0:
        print("Training loss and Accuracy after epoch {:02d}: {:.3f} and {:.8f}".format(epoch, loss_train,  accuracy))
        
print("DBN and Softmax Layer training Completed!")


#--------------------------------------------------------------------------------------
# 검정 데이터에 대해서  Accuray를 구한다
#--------------------------------------------------------------------------------------
test_act_h2, _ = cd(x_test.astype('float32'))
test_logits = tf.nn.softmax(tf.matmul(test_act_h2,w_c) + b_c, 1)
test_correct_pred = tf.equal(tf.argmax(test_logits,1),tf.argmax(x_test,1))
test_accuracy = tf.reduce_mean(tf.cast(test_correct_pred,tf.float32))

print("Accuracy for test data:  {:.8f}".format(accuracy))  #0.89843750


"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)
예제 2-3. Iris 데이터에 대한 Autoencoder

피셔의 Iris 데이터에 은닉노드 2개를 가진 선형 Autoencoder를 적용하여 2차원 데이터로
축약하고 축약된 데이터의 산점도를 통해 각 관측값이 어떤 그룹에 속하는지를 시각적으로
확인한다.
"""

## 필요한 라이브러리를 불러들임
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## iris 데이터 불러오기
url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
iris = pd.read_csv(url)

## iris 데이터를 입력 데이터와 출력 데이터로 분리
irisx = np.array(iris.iloc[:,:4])
irisy = iris.iloc[:,4]

## 입력 데이터의 min-max 정규화
minmax = np.amin(irisx,0),np.amax(irisx,0)
no_irisx = (irisx-minmax[0])/(minmax[1]-minmax[0])

## 학습 관련 매개변수 설정
n_input = 4
n_hidden = 2
n_output = n_input
learning_rate = 0.01
n_class = 3
num_epoch = 1000

## 오토인코더 구축 및 계산
x = tf.keras.Input(shape=(n_input,))
hidden = tf.keras.layers.Dense(n_hidden)(x)
output = tf.keras.layers.Dense(n_output)(hidden)

autoencoder = tf.keras.Model(x,output)
encoder = tf.keras.Model(x,hidden)

hidden_x = tf.keras.Input(shape=(n_hidden,))
decoder_layer = autoencoder.layers[-1]
decoder = tf.keras.Model(hidden_x,decoder_layer(hidden_x))

## Mean Square Error (MSE) loss funtion, Adam optimizer
autoencoder.compile(loss='mean_squared_error', optimizer='Adam')

## train the autoencoder
autoencoder.fit(no_irisx,no_irisx,epochs=num_epoch)

codings_val = encoder.predict(no_irisx)

## 산점도 그리기
plt.scatter(codings_val[np.where(irisy=='setosa')[0],0],
                        codings_val[np.where(irisy=='setosa')[0],1],color='red')
plt.scatter(codings_val[np.where(irisy=='virginica')[0],0],
                        codings_val[np.where(irisy=='virginica')[0],1],color='blue')
plt.scatter(codings_val[np.where(irisy=='versicolor')[0],0],
                        codings_val[np.where(irisy=='versicolor')[0],1],color='black')

plt.xlabel('$z_1$',fontsize=16)
plt.xlabel('$z_2$',fontsize=16,rotation=0)
plt.show()


"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)
예제 2-4. MNIST 데이터에 대한 Denoising Autoencoder

훈련용 60,000개와 검정용 10,000개로 이루어진 MNIST 데이터에 은닉노드 1,024개를 가
진 잡음 제거 오토인코더를 적용하여 8개의 검정용 이미지와 복원된 이미지를 비교한다. 정
규분포 잡음의 경우에 대해서만 설명한다.
"""

## 필요한 라이브러리를 불러들임
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import time

start_time = time.time()

## MNIST 데이터를 읽어들임
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

## 학습 관련 매개변수 설정
learning_rate = 0.01
batch_size = 150
display_step = 1
examples_to_show = 10
noise_level = 1.0

n_input = 784
n_hidden = 300

## 잡음 포함 입력, 가중치 및 편향을 정의함
x_train_noisy = x_train+noise_level*tf.random.normal(tf.shape(x_train))
x_test_noisy = x_test+noise_level*tf.random.normal(tf.shape(x_test))
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

## 오토인코더 구축 및 계산
x = tf.keras.Input(shape=(n_input,))
encoded = tf.keras.layers.Dense(n_hidden,use_bias=True)(x) 
decoded = tf.keras.layers.Dense(n_input,use_bias=True)(encoded)

autoencoder = tf.keras.Model(x,decoded)

# 분리된 Encoder 모델
encoder = tf.keras.Model(x,encoded)

# Decoder 모델
encoded_input = tf.keras.Input(shape=(n_hidden,))
decoder_layer = autoencoder.layers[-1]
decoder = tf.keras.Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error',optimizer='Adam')

num_epoch = 100
num_batch = int(len(x_train)/batch_size)
autoencoder.fit(x_train_noisy,x_train,                
                epochs=num_epoch,
                batch_size=num_batch,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

pred = autoencoder.predict(x_test[:8])
fig, a = plt.subplots(2,8,figsize=(10,2))
for i in range(8):
    a[0][i].imshow(np.reshape(x_test[i],(28,28)),cmap='gray')
    a[1][i].imshow(np.reshape(pred[i],(28,28)),cmap='gray')

## 가중치 저장
w_encod = encoder.get_weights()[0]
w_decod = decoder.get_weights()[0]

end_time = time.time()

print('elapsed time:',end_time - start_time)


"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)

예제 2-5. Autoencoder로 Pre-training 한 후에 역전파 알고리즘으로 Fine-tuning 하는 DNN 모형

UCI 기계학습 저장소의 온라인 뉴스 인기도 데이터(online news popularity dataset)는 
60개의 입력변수와 1개의 출력변수 shares에 대한 관측값 39,644개로 이루어져 있다. 
먼저 출력변수 shares의 중앙값을 기준으로 전체 데이터를 두 개의 그룹으로 나누고, 
2:1 비율로 랜덤하게 훈련 데이터와 검정 데이터로 나눈다. 이때 전체 데이터에서의 
두 그룹의 관측값 개수 비율이 훈련 데이터에서도 같도록 만드시오. 오토인코더 기반 
예비훈련을 사용하는 심층 신경망을 활용하여 검정 데이터에 가장 좋은 정확도와 
AUCarea under ROC curve값을 제공하는 심층 신경망의 구조를 찾고, 정확도, AUC값과 
함께 설명하시오.
"""

# 절대 임포트 설정
from __future__ import division, print_function, absolute_import
## 필요한 라이브러리를 불러들임
import tensorflow as tf
#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

tf.compat.v1.reset_default_graph #reset the graph

## 데이터 불러오기
online = pd.read_csv("OnlineNewsPopularity.csv",header=0)      #39644*61

online.iloc[:,60].describe()

pd.DataFrame(np.where(online.iloc[:,60]>=1400,1,0))

online = pd.concat([online,pd.DataFrame(np.where(online.iloc[:,60]>=1400,1,0))],axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(online.iloc[:,1:60])
online_s = scaler.transform(online.iloc[:,1:60])

x_train, x_test, y_train, y_test = train_test_split(online_s,
                                                    online.iloc[:,61],test_size=0.33, random_state=12050163,stratify=online.iloc[:,61])

#x_train = tf.cast(x_train, tf.float32)
#x_test = tf.cast(x_test, tf.float32)

#y_train_1hot = pd.get_dummies(y_train)
#y_test_1hot = pd.get_dummies(y_test)
y_train_1hot = tf.one_hot(y_train,depth = 2)
y_test_1hot = tf.one_hot(y_test,depth = 2)

# DNN 매개변수
n_input = 59         # DNN의 입력층의 노드 개수
n_hidden1 = 40       # DNN의 첫번째 은닉층의 노드 개수 
n_hidden2 = 30       # DNN의 두번째 은닉층의 노드 개수 

#===========================================================================================#
##(1)======= 오토인코더를 이용하여 2개의 은닉층을 갖는 심층 신경망을 Pre-training 하기 ========##
#===========================================================================================#

# 학습 매개변수
learning_rate_ae = 0.01
n_epoch_ae = 200     # pre-training 에포크 횟수 (iteration)
batch_size_ae = 128          

# Greedy Layer-wise pre-training을 위한 AE 매개변수
n_input1_ae = 59         # 첫번째 오토인코더의 입력층의 노드 개수
n_hidden1_ae = 40        # 첫번째 오토인코더의 은닉층의 노드 개수 
n_input2_ae = 40         # 첫번째 오토인코더의 입력층의 노드 개수 (= 첫번째 오토인코더의 은닉층의 노드 개수)
n_hidden2_ae = 30        # 첫번째 오토인코더의 은닉층의 노드 개수 

# 가중치 및 편향을 정의함
weights1 = {
    'encoder1_h': tf.Variable(tf.random.normal([n_input1_ae, n_hidden1_ae])),
    'decoder1_h': tf.Variable(tf.random.normal([n_hidden1_ae, n_input1_ae])),
}
biases1 = {
    'encoder1_b': tf.Variable(tf.random.normal([n_hidden1_ae])),
    'decoder1_b': tf.Variable(tf.random.normal([n_input1_ae])),
}

weights2 = {
    'encoder2_h': tf.Variable(tf.random.normal([n_input2_ae, n_hidden2_ae])),
    'decoder2_h': tf.Variable(tf.random.normal([n_hidden2_ae, n_input2_ae])),
}
biases2 = {
    'encoder2_b': tf.Variable(tf.random.normal([n_hidden2_ae])),
    'decoder2_b': tf.Variable(tf.random.normal([n_input2_ae])),
}

# 가중치와 편향의 이름을 변경 
w1_e1 = weights1['encoder1_h']
b1_e1 = biases1['encoder1_b']
w2_e2 = weights2['encoder2_h']
b2_e2 = biases2['encoder2_b']

w1_d1 = weights1['decoder1_h']
b1_d1 = biases1['decoder1_b']
w2_d2 = weights2['decoder2_h']
b2_d2 = biases2['decoder2_b']

# 2개의 인코더를 구축함
def encoder1(x):
    enc1_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, w1_e1),b1_e1))
    return enc1_layer

def encoder2(x):
    enc2_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, w2_e2),b2_e2))
    return enc2_layer

# 2개의 디코더를 구축함
def decoder1(x):
    # Decoder Hidden layer with sigmoid activation #1
    dec1_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, w1_d1),b1_d1))
    return dec1_layer

def decoder2(x):
    # Decoder Hidden layer with sigmoid activation #1
    dec2_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, w2_d2),b2_d2))
    return dec2_layer

# 첫번째 Autoencoder의 관련 가중치와 편향을 구하기 위한 손실함수 계산
def loss1(x):
    encoder1_op = encoder1(x)
    decoder1_op = decoder1(encoder1_op)
    recon_loss1 = tf.reduce_mean(tf.square(x - decoder1_op))
    return recon_loss1   # reconstruction loss 계산

# 첫번째 Autoencoder의 관련 가중치와 편향을 구하기 위한 손실함수 계산
def loss2(x):
    encoder1_op = encoder1(x)
    encoder2_op = encoder2(encoder1_op)
    decoder2_op = decoder2(encoder2_op)
    recon_loss2 = tf.reduce_mean(tf.square(encoder1_op - decoder2_op))
    return recon_loss2   # reconstruction loss 계산

# 첫번째 Autoencoder의 관련 가중치와 편향을 구하기 위한 위한 경사 (즉, 기울기) 계산
def gradient1(x, w1_e1, b1_e1):
    with tf.GradientTape() as tape:
        loss_value1 = loss1(x)
    return tape.gradient(loss_value1, [w1_e1, b1_e1])

# 두번째 Autoencoder의 관련 가중치와 편향을 구하기 위한 위한 경사 (즉, 기울기) 계산
def gradient2(x, w2_e2, b2_e2):
    with tf.GradientTape() as tape:
        loss_value2 = loss2(x)
    return tape.gradient(loss_value2, [w2_e2, b2_e2])

"""
# 데이터를 배치크기로 자르는 함수 만들기 (우리는 필요 없음)
def next_batch(batch_size,x_train,y_train,index_in_epoch,epoch_completed):

    #global x_train
    #global y_train1
    #global index_in_epoch
    #global epoch_completed

    start = index_in_epoch
    index_in_epoch += batch_size_ae

    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > n_examples:
        # finished epoch
        epoch_completed += 1
        # shuffle the data
        perm = np.arange(n_examples)
        np.random.shuffle(perm)
        x_train = x_train[perm]
        y_train = y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size_ae
        assert batch_size_ae <= n_examples
    end = index_in_epoch
    return x_train[start:end], y_train[start:end]

"""

##--------------- 실제로 DNN을 Pre-training 하기 --------------------##
total_batch_ae = int(len(x_train)/batch_size_ae)
#n_epoch_ae = 200    # pre-training 에포크 횟수 (iteration)
#batch_size_ae = 128
#learning_rate_ae = 0.01

display_step = 2

optimizer1 = tf.keras.optimizers.RMSprop(learning_rate_ae)
optimizer2 = tf.keras.optimizers.RMSprop(learning_rate_ae)

for epoch in range(n_epoch_ae):
    for i in range(total_batch_ae):
        batch_xs, batch_ys = x_train[i*batch_size_ae:(i+1)*batch_size_ae],y_train_1hot[i*batch_size_ae:(i+1)*batch_size_ae]
                       
        # DNN 관련 가중치 및 편향 w1_e1, b1_e1, w2_e2, b2_e2 업데이트
        del_w1_e1, del_b1_e1 = gradient1(batch_xs.astype('float32'), w1_e1, b1_e1)  
        change_w1_e1 = del_w1_e1 * learning_rate_ae   # adjustment amount for weight
        change_b1_e1 = del_b1_e1 * learning_rate_ae   # adjustment amount for bias
        w1_e1.assign_sub(change_w1_e1)           # subract from w1_e1
        b1_e1.assign_sub(change_b1_e1)           # subract from b1_e1
        
        del_w2_e2, del_b2_e2 = gradient2(batch_xs.astype('float32'), w2_e2, b2_e2)  
        change_w2_e2 = del_w2_e2 * learning_rate_ae   # adjustment amount for weight
        change_b2_e2 = del_b2_e2 * learning_rate_ae   # adjustment amount for bias
        w2_e2.assign_sub(change_w2_e2)           # subract from w2_e2
        b2_e2.assign_sub(change_b2_e2)           # subract from b2_e2
        
        loss1_train = loss1(batch_xs.astype('float32'))
        loss2_train = loss2(batch_xs.astype('float32'))
        
        total_loss = loss1_train + loss2_train
        
    if epoch % display_step == 0:
        print("Training losses after epoch {:02d}: {:.6f} ".format(epoch, total_loss))
        
print("DNN Greedy Layer-wise Pre-training Completed!")


#===============================================================================================#
##(2)======= 역전파 알고리즘을 이용하여 2개의 은닉층을 갖는 심층 신경망을 fine-tuning 하기 ========##
#===============================================================================================#
from __future__ import print_function

## 데이터 불러오기
online = pd.read_csv("OnlineNewsPopularity.csv",header=0)

online.iloc[:,60].describe()

pd.DataFrame(np.where(online.iloc[:,60]>=1400,1,0))

online = pd.concat([online,pd.DataFrame(np.where(online.iloc[:,60]>=1400,1,0))],axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(online.iloc[:,1:60])
online_s = scaler.transform(online.iloc[:,1:60])

x_train, x_test, y_train, y_test = train_test_split(online_s, online.iloc[:,61],test_size=0.33, random_state=12050163,stratify=online.iloc[:,61])

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

#y_train_1hot = pd.get_dummies(y_train)
#y_test_1hot = pd.get_dummies(y_test)
y_train_1hot = tf.one_hot(y_train,depth = 2)
y_test_1hot = tf.one_hot(y_test,depth = 2)

# DNN 학습 매개변수
n_input = 59         # DNN의 입력층의 노드 개수
n_hidden1 = 40       # DNN의 첫번째 은닉층의 노드 개수 
n_hidden2 = 30       # DNN의 두번째 은닉층의 노드 개수 
n_class = 2         # DNN의 두번째 출력층의 노드 개수 

lr_rate = 0.01

## 신경망 매개변수 (가중치, 편의) 설정 및 초기화
w1 = tf.Variable(w1_e1, name='weights1')
b1 = tf.Variable(b1_e1, name='biases1')

w2 = tf.Variable(w2_e2, name='weights2')
b2 = tf.Variable(b2_e2,name='biases2')

wo = tf.Variable(tf.random.normal([n_hidden2, n_class], mean=0, stddev=1/np.sqrt(n_input)), name='weightsOut')
bo = tf.Variable(tf.random.normal([n_class], mean=0, stddev=1/np.sqrt(n_input)), name='biasesOut')

## 가중치 및 편향을 하나로 묵음 
variables = [w1, b1, w2, b2, wo, bo]  

## 학습 관련 함수 정의하기  
def feed_forward(x):
    # layer1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
    # layer2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, w2), b2))
    # output layer
    output = tf.nn.softmax(tf.add(tf.matmul(layer2, wo), bo))
    return output

def loss_fn(y_pred, y_true):
#    loss = tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[1]))
    return loss

def acc_fn(y_pred, y_true):
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    y_true = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
    predictions = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    return tf.reduce_mean(predictions)

def backward_prop(batch_xs, batch_ys):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    with tf.GradientTape() as tape:
        predicted = feed_forward(batch_xs)
        step_loss = loss_fn(predicted, batch_ys)
    grads = tape.gradient(step_loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

## 신경망을 학습하는 과정
n_epochs = 50
batch_size = 128
total_batch = int(len(x_train)/batch_size)
n_shape = x_train.shape[0]   #또는 len(x_train) 사용
no_steps = n_shape//batch_size
display_step = 2

for epoch in range(n_epochs):
    avg_loss = 0.
    avg_acc = 0.
    for i in range(total_batch):
        batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size],y_train_1hot[i*batch_size:(i+1)*batch_size]
        
        pred_ys = feed_forward(batch_xs)
        avg_loss += float(loss_fn(pred_ys, batch_ys)/no_steps) 
        avg_acc += float(acc_fn(pred_ys, batch_ys) /no_steps)
        backward_prop(batch_xs, batch_ys)

    if epoch % display_step == 0:        
        #print('Epoch: {epoch}, Training Loss: {avg_loss}, Training ACC: {avg_acc}')
        print("Training loss and Accuracy after epoch {:02d}: {:.4f} and {:.8f}".format(epoch, avg_loss, avg_acc))
        #Training loss and Accuracy after epoch 48: 0.5157 and 0.74818841

print("Neural Network Training Completed!")        

## 검정 데이터에 대해서  싱경망의 Accuray를 구함
test_pred_y = feed_forward(x_test) 
test_accuracy = acc_fn(test_pred_y, y_test_1hot)

print("Accuracy for test data:  {:.8f}".format(test_accuracy))  #Accuracy for test data:  0.62088203


"""
참고논문: Auto-encoding variational bayes (D.P. Kingma and M. Welling, 2014)

예제 2-6. MNIST 데이터 대해 다층 신경망을 사용하는 VAE 예제
"""

# 절대 임포트 설정
from __future__ import division, print_function, absolute_import

## 필요한 라이브러리를 불러들임
import tensorflow as tf
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    # K is the keras backend
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 20

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
#    plot_model(vae,
#               to_file='vae_mlp.png',
#               show_shapes=True)

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")


"""
참고교재: Auto-encoding variational bayes (D.P. Kingma and M. Welling, 2014)

예제 2-7. MNIST 데이터 대해 다층 신경망을 사용하는 AAE 예제

https://raw.githubusercontent.com/deepgradient/adversarial-autoencoder/master/unsupervised_aae_deterministic.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path.cwd()

# -------------------------------------------------------------------------------------------------------------
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

# -------------------------------------------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'outputs'
output_dir.mkdir(exist_ok=True)

experiment_dir = output_dir / 'unsupervised_aae_deterministic'
experiment_dir.mkdir(exist_ok=True)

latent_space_dir = experiment_dir / 'latent_space'
latent_space_dir.mkdir(exist_ok=True)

reconstruction_dir = experiment_dir / 'reconstruction'
reconstruction_dir.mkdir(exist_ok=True)

sampling_dir = experiment_dir / 'sampling'
sampling_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------------------------------------------------------
# Loading data
print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the dataset
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator
batch_size = 256
train_buf = 60000

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------
# Create models
image_size = 784
h_dim = 1000
z_dim = 2


def make_encoder_model():
    inputs = tf.keras.Input(shape=(image_size,))
    x = tf.keras.layers.Dense(h_dim)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    encoded = tf.keras.layers.Dense(z_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=encoded)
    return model


def make_decoder_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    reconstruction = tf.keras.layers.Dense(image_size, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model():
    encoded = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(h_dim)(encoded)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(h_dim)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    prediction = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    return model


encoder = make_encoder_model()
decoder = make_decoder_model()
discriminator = make_discriminator_model()

# -------------------------------------------------------------------------------------------------------------
# Define loss functions
ae_loss_weight = 1.
gen_loss_weight = 1.
dc_loss_weight = 1.

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
accuracy = tf.keras.metrics.BinaryAccuracy()


def autoencoder_loss(inputs, reconstruction, loss_weight):
    return loss_weight * mse(inputs, reconstruction)


def discriminator_loss(real_output, fake_output, loss_weight):
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_weight * (loss_fake + loss_real)


def generator_loss(fake_output, loss_weight):
    return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)


# -------------------------------------------------------------------------------------------------------------
# Define cyclic learning rate
base_lr = 0.00025
max_lr = 0.0025

n_samples = 60000
step_size = 2 * np.ceil(n_samples / batch_size)
global_step = 0

# -------------------------------------------------------------------------------------------------------------
# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)


# -------------------------------------------------------------------------------------------------------------
# Training function
@tf.function
def train_step(batch_x):
    # -------------------------------------------------------------------------------------------------------------
    # Autoencoder
    with tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch_x, training=True)
        decoder_output = decoder(encoder_output, training=True)

        # Autoencoder loss
        ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

    ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Discriminator
    with tf.GradientTape() as dc_tape:
        real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
        encoder_output = encoder(batch_x, training=True)

        dc_real = discriminator(real_distribution, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Discriminator Loss
        dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)

        # Discriminator Acc
        dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                          tf.concat([dc_real, dc_fake], axis=0))

    dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

    # -------------------------------------------------------------------------------------------------------------
    # Generator (Encoder)
    with tf.GradientTape() as gen_tape:
        encoder_output = encoder(batch_x, training=True)
        dc_fake = discriminator(encoder_output, training=True)

        # Generator loss
        gen_loss = generator_loss(dc_fake, gen_loss_weight)

    gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

    return ae_loss, dc_loss, dc_acc, gen_loss


# -------------------------------------------------------------------------------------------------------------
# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    start = time.time()

    # Learning rate schedule
    if epoch in [60, 100, 300]:
        base_lr = base_lr / 2
        max_lr = max_lr / 2
        step_size = step_size / 2

        print('learning rate changed!')

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    for batch, (batch_x) in enumerate(train_dataset):
        # -------------------------------------------------------------------------------------------------------------
        # Calculate cyclic learning rate
        global_step = global_step + 1
        cycle = np.floor(1 + global_step / (2 * step_size))
        x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
        clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr)
        ae_optimizer.lr = clr
        dc_optimizer.lr = clr
        gen_optimizer.lr = clr

        ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x)

        epoch_ae_loss_avg(ae_loss)
        epoch_dc_loss_avg(dc_loss)
        epoch_dc_acc_avg(dc_acc)
        epoch_gen_loss_avg(gen_loss)

    epoch_time = time.time() - start
    print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
          .format(epoch, epoch_time,
                  epoch_time * (n_epochs - epoch),
                  epoch_ae_loss_avg.result(),
                  epoch_dc_loss_avg.result(),
                  epoch_dc_acc_avg.result(),
                  epoch_gen_loss_avg.result()))

    # -------------------------------------------------------------------------------------------------------------
    if epoch % 10 == 0:
        # Latent space of test set
        x_test_encoded = encoder(x_test, training=False)
        label_list = list(y_test)

        fig = plt.figure()
        classes = set(label_list)
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        kwargs = {'alpha': 0.8, 'c': [colormap[i] for i in label_list]}
        ax = plt.subplot(111, aspect='equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles = [mpatches.Circle((0, 0), label=class_, color=colormap[i])
                   for i, class_ in enumerate(classes)]
        ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45), fancybox=True, loc='center left')
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s=2, **kwargs)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # Reconstruction
        n_digits = 20  # how many digits we will display
        x_test_decoded = decoder(encoder(x_test[:n_digits], training=False), training=False)
        x_test_decoded = np.reshape(x_test_decoded, [-1, 28, 28]) * 255
        fig = plt.figure(figsize=(20, 4))
        for i in range(n_digits):
            # display original
            ax = plt.subplot(2, n_digits, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n_digits, i + 1 + n_digits)
            plt.imshow(x_test_decoded[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))
        plt.close('all')

        # Sampling
        x_points = np.linspace(-3, 3, 20).astype(np.float32)
        y_points = np.linspace(-3, 3, 20).astype(np.float32)

        nx, ny = len(x_points), len(y_points)
        plt.subplot()
        gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

        for i, g in enumerate(gs):
            z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
            z = np.reshape(z, (1, 2))
            x = decoder(z, training=False).numpy()
            ax = plt.subplot(g)
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.savefig(sampling_dir / ('epoch_%d.png' % epoch))
        plt.close('all')







