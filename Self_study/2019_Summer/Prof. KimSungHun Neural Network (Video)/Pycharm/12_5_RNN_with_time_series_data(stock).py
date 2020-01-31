#RNN은 time series에서도 연결된다.
#이는 many to one의 연결고리와 같다. (이에 대한 내용은 RNN txt참고)

'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os


tf.set_random_seed(777)  # reproducibility

# 이 아래부분 그냥하지말기. plot이 안뜸.
# if "DISPLAY" not in os.environ:
#     #os.environ 는 현재 환경변수를 출력하라는 것.
#     #기본 경로들임. 파일의 접근을 쉽고 편하게 하기 위해 환경변수 설정을 함.
#     #환경변수에 대한 설명 :
#     #https://c-calliy.tistory.com/42
#
#     #나는 현재의 환경변수에 DISPLAY가 없다.
#
#     # remove Travis CI Error
#     matplotlib.use('Agg')
#     #뭔소릴까... 궁금하면 천천히 찾아보기. 일단 pass
#     # https://stackoverflow.com/questions/44086597/cant-use-matplotlib-useagg-graphs-always-show-on-the-screen
#     # 이게 참고가 될까? 나중에 읽어보기.

import matplotlib.pyplot as plt

#최소,최대를 0,1로 놓고 비율로 놓는 변환. 열별로 최소, 최대를 계산함.
def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0) #axis=0 (열) , 가장 바깥쪽임. axis=-1의 사용 이유를 생각하자.
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
# data 02인 주식 데이터를 가지고 생각해보자.
# os.getcwd()
os.chdir("C:/Users/82104/Desktop/")
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',') #  앞에 '#'으로 되어있는 주석부분은 안읽고 넘긴다.
xy = xy[::-1]  # reverse order (chronically ordered)
#가장 위의 데이터가 최근인가봄?

# train/test split
train_size = int(len(xy) * 0.7) #대략 70%만 쓰자.
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

# build datasets
#batch_size만큼 묶음.
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        #나루 하루짜리를 Close 가격으로 예측하는거임.
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)
#12_4에서 했다.

trainX, trainY = build_dataset(train_set, seq_length) #512 - 7 = 505 번의 loop를 돈다..
testX, testY = build_dataset(test_set, seq_length)

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
# [batch , sequence length, data size]
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
# hidden_dim을 주고, fully_connected를 만들어서 사용함.


outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions))) #예측이 얼마나 잘됐는지

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX}) #예측값과 관련된 부분.
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY) #파란선
    plt.plot(test_predict) #주황선
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

# LSTM 과 선형회귀와 어떤 차이가 있는지 살펴보기
# LSTM은 LONG Short-term Memory의 약자임.
# 끝으로 RNN은 다양하게 쓰일 수 있으므로 반드시 알아놓고 가기


