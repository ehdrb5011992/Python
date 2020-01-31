# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
#set이라는 함수는 r의 unique 함수와 같음.
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex
#이 표현 꼭 주목해서 보기!!

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
# 박스의 개수라고 봐도 무방. 또는 x데이터의 길이. 우리는 한개씩 뺄거기 때문에 길이를 -1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello
#이렇게 한줄 밀린걸로 학습시키는거임. 다음에 무슨단어가 나올지를 학습하는 것이기 때문.

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data // 정수형임에 주목.
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label // 정수형임에 주목.

x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
#X를 num_classes 개수만큼 one-hot시켜라.
#shape에 변화가 있는 one_hot임. 항상 shape을 체크하고 원하는 shape인지 볼것.
#이전의 softmax에서 배울땐, Y값을 one_hot encoding을 했었다.
#그렇기에, reshape를 해줬지만, 이번의 경우는 X데이터를 one_hot 해주기 때문에
#행렬 그 자체로 받아들여도 무방해서 one_hot encoding이 필요 없다.

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
# 곧 사라질 BasicLSTMCell 과 dynamic_rnn
# 이렇게 적용된 rnn이 나온다.
# 이제 Fully connected를 적용하자.

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

#결과 가지고 채점 시작.
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss( logits=outputs, targets=Y, weights=weights)
# sequence_loss는 rnn에서 쓰는 loss 함수. 참값의 경우로 비교한다.
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))

#운이 좋으면 잘 맞춘 문자열을 볼 수 있다. 학습이 이루어지기는 함.
#어떤 문자열이라도 입력을 처리할 수 있도록 만들었다.




#################### 살펴보기 ####################

#RNN 대신 softamx를 단순하게 써보겠다.
# Lab 12 Character Sequence Softmax only
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

##### 이 위까지 같다. #####

# flatten the data (ignore batches for now). No effect if the batch size is 1
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
#X_one_hot의 shape은 (?,15,10) 이다. 15는 살펴볼 데이터의 수임. (sequence length)
#10은 one_hot encoding되는것.
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size])
#15가 데이터의 수기에, 위처럼 reshape를 해도 의미는 바뀌지 않음.
#또한, 이렇게해야 행렬 계산이 원활함.

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
#그런데, softmax를 안씀.
# 이유는 sequence_loss에 softmax 함수가 내장되어 있기 때문.
# 아래의 사이트 참고. https://coolingoff.tistory.com/54 (12_4로 이어진다.)

#채점 시작.
# expend the data (revive the batches)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])

# Compute sequence cost/loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)  # mean all sequence loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))

#RNN을 안쓰니, 학습이 덜 되고 잘 되지도 않는다.
#그래도 좀 어느정도는 나오고는 있다.
