#고급과정 // 데이터는 이렇게 주어짐.
#softmax는 각 class를 확률로 변환시켜주는 함수임.
# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('/Users/82104/Desktop/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)


nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6 여기서부터 벌써 차이가난다.
                                         # 일단 우리가 가진 데이터는 1줄짜리임.
#주의!!!) Y의 타입을 int로 줘서 설계행렬은 0과 1뿐임을 확실히 명시!!

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot 만드는법. 꼭 숙지. tf.one_hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, shape= [-1, nb_classes]) #이 shape은 None대신 -1임!
# 반드시 reshape를 해야함. (-1은 행렬에서 임의의라는 뜻)
# rank가 1개 늘어나기 때문.
# 늘어나는 이유 : 원래 데이터는 (?,1) 차원 행렬이었고, 열 차원인 1이 7개로 나눠지면서 (1,7)행렬이 된것.
# one hot shape = (?,1,7)  -> (?,7) 로 해줘야한다. design matrix 늘 생각.
# 어디까지나 Y_one_hot도 그릇임.

print("reshape one_hot:", Y_one_hot)
# 즉, one_hot과 reshape는 따라다닌다고 생각.

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
# x입력이 16개, 출력은 y가 7개이기 때문에 7
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
# 출력의 개수와 똑같음.

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b  #logit = Xbeta + epsilon임. (우리가 생각한모형)
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss  <--- 학습시작.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
# Q) // tf.nn.softmax_cross_entropy_with_logits 말고 tf.nn.softmax_cross_entropy_with_logits_v2 쓰는 이유는
# label이 서로다른 네트워크에서 오는 경우 tf.nn.softmax_cross_entropy_with_logits 얘는 애러가 날 수 있다나?
# 대표적인 예로 GAN 의 경우가 그렇다나..? 그래서 보안책으로 tf.nn.softmax_cross_entropy_with_logits_v2가
# 나온거란다..... 나중에 배워보고 일단은 받아들이기//

# Q) // stop_gradient 역시 이것도 왜 해야하는지 모르겠다... 지식수준을 벗어남. 일단 받아들이기.

# softmax_cross_entropy_with_logits_v2 는 logits과 labels을 옵션으로 받는데,
# logits는 XW+b에 해당하는값, labels는 Y에 해당하는 값을 주면 된다.

# labels 은 tf.stop_gradient
# cost = tf.reduce_mean(  -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)   ) 얘는 Y값이 설계행렬로
# 이미 주어진 경우에나 가능하다. 보통 데이터를 받아들여서 사용하는 경우는 위처럼 해야함.
# -tf.reduce_sum(Y * tf.log(hypothesis)) 이 텀을 cross entropy라 하는 듯 하다. 강의의 D함수.
# D함수를 reduce_mean한다는 것을 잊지말기! (cross_entropy를 reduce_mean 하는것 잊지말기.)

#이렇게 설계를 함수함. (logits을 반드시 기억.)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1) #확률에서 가장 높은걸 목적인 class로 할당.
#hypothesis는 계속 갱신될 것이므로, 이렇게 시행전에 선언해도 됨.

correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
#tf.argmax(Y_one_hot, 1)를 하면 설계행렬의 각 행에서 1이 찍혀있는 열의 index를 반환해준다.
#즉, prediction과 tf.argmax(Y_one_hot,1)은 둘다 n차원 벡터이다. equal은 원소별로 같은지 비교함.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #전부 그래프만 만드는 행위.

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001): #데이터 학습을 2001번시킴. 즉 Gradient Descent Algorithm이 2001번 돌아감.
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

            #{}안에 들어오는 문법에 대해 알아놓자. 이때 :를 반드시 써줘야 함을 명심하자.
            # \t는 간격을 텝만큼 띄우라는 명령어이다.
            # 소숫점을 쓰려면 .과 f를 써줘야 하며, (f는 float의 약자)
            # %을 쓰면 비율로 값을 돌려준다.

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape // #tf.squeeze(대상)와 같음.
    for p, y in zip(pred, y_data.flatten()):
        # [[0],[1]] -> [0,1] 로 바꾸는게 flatten() (array를 벡터로나열) , 이래야 zip으로 묶고 matching이 가능.
        # 주의! matrix 타입을 벡터로 나열하는건 아님!!
        # 이는 pandas로 작업을 하다보면 깨달음.
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    # 같이 돌릴 때는 zip함수로 묶은 후 돌릴 것.
    # 재밌는 표현 사용함. 전체적으로 익숙해지기.

########### pandas로 진행시 ##############

import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)
xy = pd.read_csv('/Users/82104/Desktop/data-04-zoo.csv',delimiter=',',dtype=np.float32
                 ,skiprows = range(19) ,header = None ) #skiprows는 생략할 행 설정. 엑셀열어서 확인함.

x_data = xy.iloc[:,0:-1]
y_data = xy.iloc[:,[-1]]

x = tf.placeholder(tf.float32, shape = [None,16])
y = tf.placeholder(tf.int32, shape = [None,1])
labels=7

y_one_hot = tf.one_hot(y,labels)
y_one_hot = tf.reshape(y_one_hot,shape=[-1,labels])

W = tf.Variable(tf.random_normal([16,labels]),dtype = tf.float32,name='weight')
b = tf.Variable(tf.random_normal([labels]),dtype=tf.float32,name='bias')

logits = tf.matmul(x,W) + b
f=tf.nn.softmax(logits)

cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                 labels=tf.stop_gradient([y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

prediction = tf.argmax(f,axis=1) # n x 1
correct_pred = tf.equal(prediction,tf.argmax(y_one_hot,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :

    _, cost_val , acc_val = sess.run([train,cost,accuracy],
                                     feed_dict={x:x_data , y:y_data})
    if step % 200 == 0:
        print("step: {:5}\tcost: {:.3f} \taccuracy: {:.2%}".format(step,cost_val,acc_val))

pred = sess.run(prediction,feed_dict={x:x_data})
y_arr=np.array(y_data) # 이부분에서 다름을 반드시 명심하자!!
                       # 우리는 아직까지 np.matrix를 써 본적이 없고
                       # 앞으로도 쓸 일이 없다!!
for p , y in zip(pred, y_arr.flatten()): #tf의 squeeze와 같음.
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
