# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy  as np
#import pandas as pd
tf.set_random_seed(777)  # for reproducibility
# import os
# os.chdir("/Users/82104/Desktop/")
#xy = pd.read_csv("/Users/82104/Desktop/data-01-test-score.csv", header = None) #그밖에 sep , index_col 옵션이 있음.
xy = np.loadtxt('/Users/82104/Desktop/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1] #넘파이에서 가져오면 이렇게됨.
#x_data=xy.iloc[:,0:-1] #판다스에서 가져오면 이렇게
#x_data.head()
y_data = xy[:, [-1]] #넘파이에서 가져오면 이렇게됨.
#y_data = xy.iloc[:,[-1]]
#y_data.head()

# Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape) #넘파이 매서드
print(y_data, "\ny_data shape:", y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_data, Y: y_data})
        if step % 400 == 0:
            print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)


#

# Lab 4 Multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data

import tensorflow as tf
import os
os.chdir("/Users/82104/Desktop/") #미리 경로설정.
tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')
#파일들을 쌓아서 불러올수 있음.
#파일들의 리스트를 처음에 받음. 이경우 1개임.

reader = tf.TextLineReader() #텍스트를 읽을꺼임.
key, value = reader.read(filename_queue) #텍스트 파일을 읽을때 일반적으로 사용.
#참고하면 더 쉽다. https://bcho.tistory.com/1165

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]] #floating point 정의
xy = tf.decode_csv(value, record_defaults=record_defaults)
#value를 csv로 decode 해라 라는 뜻. 는 찾아보기

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10) #10개씩 가져옴.
sess=tf.Session()
#x_batch  = sess.run([train_x_batch]) 얘를 불러오면 렉먹음;
#일종의 펌프같은 역할. batch에 대해서도 찾아보기.
#여기까지가 불러오는 방법임.


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    #펌프질 해서 데이터를 가져옴.
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    #가져온 데이터를 feed로 줌.
    if step % 200 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
#복잡해보이지만 이대로 따라하면 됨. x_batch , y_batch 를 씀에 주의.

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ", #내 점수 예측
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", #친구들 점수 예측
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
