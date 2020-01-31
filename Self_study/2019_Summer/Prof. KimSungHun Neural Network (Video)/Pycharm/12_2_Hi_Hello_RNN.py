# Lab 12 RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

idx2char = ['h', 'i', 'e', 'l', 'o'] #숫자는 각 인덱스에 해당.
# Teach hello: hihell -> ihello
# RNN 의 데이터를 가공하는게 어려워 보일 수 있다. 그러나 간단함!
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell 가 왼쪽처럼 주어져 있으면,
#아래처럼 one_hot_encoding을 해야함. X만 one_hot_encoding해도 됨.
#이에대한 이유는 softmax와 이어짐.
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0  // 0은 index를 의미함. 한칸민것.
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
#3개의 리스트임을 알기. <- softmax와 다른점이다. 주의!!!

y_data = [[1, 0, 2, 3, 3, 4]]    # ihello <-- True 데이터임.
#한쪽만 one_hot encoding해도 무방. 나중에 softmax와 비교해서 자세히 고민해보기.

#얘를 자유롭게 다루면 RNN을 쉽게 할 수 있는거임.

num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
#위의 size는 RNN size
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
#input_dim 때문에 float32가 필요.
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label
#sequence_length는 그 자체로 정수기 때문에 int32가 필요.

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
#이렇게 설정해도, 형태는 유지되는 체로 가로는 데이터, 세로는 각 단어 특징에 대한
#데이터 행렬이 만들어진다.
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
#위의 주석 3줄이 아래 한줄임.
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
#tf.reshape에서 outputs 을 [a,b,c]로 바꾼다는건 차원을 a X b X c 로 한다는 말.

weights = tf.ones([batch_size, sequence_length])

#weights은 1 x 6 행렬이고, 원소는 전부 1
#tf에서는 sequence_loss 라는 함수를 제공함.
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights) #weights는 1로 계산함.
#쉽게 학습을 시킬 수 있음.
#주의할 점은 RNN에서 logits을 outputs으로 바로 놓으면 사실 좀 무리가 있다.
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2) #가장 안쪽에 있는 값들 중 최대

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data}) #훈련
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
        # l, result, _ = sess.run([loss, prediction, train], feed_dict={X: x_one_hot, Y: y_data})
        # print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # 주석은 본문과 다름.
        # 본문처럼 해야 한번 돌아가서 얻어진 모형을 가지고 예측을 시행함.

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)] #차원을 늘려서, 위에서만든 리스트에 대응.
        #index에 따른 문자값을 출력.
        print("\tPrediction str: ", ''.join(result_str))

