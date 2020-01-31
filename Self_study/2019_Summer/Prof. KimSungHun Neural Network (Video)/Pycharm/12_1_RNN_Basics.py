# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://learningtensorflow.com/index.html
# http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
# input 차원은 (1,1,4)가 될 것이다.

with tf.variable_scope('one_cell') as scope:
    # One cell RNN input_dim (4) -> output_dim (2)
    hidden_size = 2 #내가 마음대로 정하는 값.
    #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size) #출력할 size를 hidden_size라고 함.
    #cell = rnn.BasicRNNCell(num_units=hidden_size) #얘와 동일.
    # tf.nn.rnn_cell.LSTMCell 도 있다. 그러나 역시 , keras쓰자.

    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size) #얘로 써야 후의 버전에서도 다룰 수 있음.
    # keras로 익히자. 위의 작업은 cell을 만드는거임.


    # tf.keras.LSTMcell 도 있음. LSTM 은 RNN에서 좀더 발전된 구성으로, deep한 경우에도 좋음.
    # 그러므로, 그냥 LSTM을 쓰자.
    print(cell.output_size, cell.state_size)

    x_data = np.array([[h]], dtype=np.float32) # x_data = [[[1,0,0,0]]] 살펴보자.
    #일부러 3겹의 list를 줌. shape의 형태를 [batch , sequence, input_size] 로 주기 위함.
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32) #두가지 출력을 냄.
    #주로 output을 많이 사용하게 됨. 위의 코드가 RNN을 시행하라는 뜻. 초기값을 알아서 정하고,
    #dynmaic_rnn을 사용함.
    #원래 dynamic_rnn은
    #여기서 output 차원은 (1,1,2)가 될 것이다.

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval()) #출력해봄.
    #pp.pprint(_states.eval()) #얘는 위와 같은값임. 알고리즘에서 처음이기때문에 그렇다.
    #다만, 차원이다름. 더군다나, _states는 1개의 셀에서 진행되고 있으므로 굳이 필요도 없는값.

    #https://sshkim.tistory.com/153 참고.

with tf.variable_scope('two_sequances') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5
    hidden_size = 2
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
    x_data = np.array([[h, e, l, l, o]], dtype=np.float64)
    #얘의 차원은 (1,5,4)이다. 이때, 5에 해당하는게 sequence_length이며,
    #이 값이 오른쪽으로 뻗어나가는 데이터들(states) 임. 즉, 오른쪽으로 뻗어나가는 박스가 5개
    print(x_data.shape)
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float64)
    #마찬가지로, dynamic_rnn도 미래에는 사라질 함수임. 나중에 따로 공부해야할듯..
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval()) # 예상하는 shape는 (1,5,2) 임. 오른쪽의 셀들로 진행 상황에 대한
    #결과 값들을 각각 뽑아내줌.
    pp.pprint(_states.eval()) # shape은 (1,2) 이며, h_1을 의미한다.(첫번째 셀의 특성)
    #즉 이때 W_hh는 shape가 (1,5,1) 이 되는것. (matmul의 연산에 의해 y차원의 결과가 (1,5,2)가 나와야함.)


#데이터를 여러개 줌으로써, batch_size 를 조절함. 데이터의 갯수임.
with tf.variable_scope('3_batches') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    hidden_size = 2
    #cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size , state_is_tuple=True) 사라진댄다.
    cell = tf.keras.layers.LSTMCell(units = hidden_size ) #keras 쓰자.
    outputs, _states = tf.nn.dynamic_rnn(
        cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    #똑같음.

with tf.variable_scope('3_batches_dynamic_length') as scope:
    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, x_data, sequence_length=[5, 3, 4], dtype=tf.float32)
    #sequence_length는 각 데이터들의 결과 길이를 몇차원으로 입력받을 것인지에 대한 값.
    #즉 입력 데이터를 가변적으로 받겠다는 뜻.
    #첫번째는 5개만 입력받음. 두번째는 3개만 입력받음. 세번째는 4개만 입력받음.
    #즉 , 5개의 sequence box중 5개, 3개, 4개만 활성화됨.

    #다음의 사이트 참고.
    #https://trendy00develope.tistory.com/47

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

with tf.variable_scope('initial_state') as scope:
    batch_size = 3
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch: 3
    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    #cell = tf.keras.layers.LSTMCell(units = hidden_size) 얘를 쓰면안됨... 아래 zero_state 라는 메서드가없음.
    initial_state = cell.zero_state(batch_size, tf.float32)
    #초기 state는 0으로 초기화. 튜플로 되어있음.
    #초기값이 필요하다면, 이렇게 만들어서 넣어주면 된다. 또는
    #initial_state = tf.random_normal(shape=(batch_size, hidden_size), mean=1.0)
    #처럼 초기값을 임의로 줄 수있다.. 그러나 위는 잘 안됨.. 나중에 고민해보기.

    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())


################## 이 아래는 연습해보기. ##################
# Create input data
batch_size=3
sequence_length=3
input_dim=5

x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
#np.arange 함수. array가 아님!
pp.pprint(x_data)  # batch, sequence_length, input_dim


with tf.variable_scope('generated_data') as scope:
    # One cell RNN input_dim (3) -> output_dim (5). sequence: 5, batch: 3
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    #이렇게 하면 3*5 짜리 행렬로 0이 채워진채 초기값이 완성됨.
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval()) #위에서 했던것과 똑같다.


with tf.variable_scope('MultiRNNCell') as scope:
    # Make rnn
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 3 layers
    #cell을 재정의해서 새롭게 층을 쌓음. 즉, 이런 코드를 사용해서 deep하게 정의할 수 있음.
    #다만 이렇게 하는경우는, input과 output의 차원이 같은경우에만 가능한듯..?

    # rnn in/out
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

with tf.variable_scope('dynamic_rnn') as scope:
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32,
                                         sequence_length=[1, 3, 2])
    # lentgh 1 for batch 1, lentgh 2 for batch 2

    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size
    #특별한거 없다.


# 양방향에서 단어를 훑는거임. 기존의 dynamic_rnn보다 훨씬 좋은 성능을 기대할 수 있다.
# 문자같은 예에서 타당한 생각이다. 이런게 있다정도로 알기.
# 참고 : https://excelsior-cjh.tistory.com/156 (그림제공)
with tf.variable_scope('bi-directional') as scope:
    # bi-directional rnn
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data,
                                                      sequence_length=[2, 3, 1],
                                                      dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(sess.run(outputs))
    pp.pprint(sess.run(states))

# flattern based softmax
hidden_size=3
sequence_length=5
batch_size=3
num_classes=5

pp.pprint(x_data) # hidden_size=3, sequence_length=4, batch_size=2
x_data = x_data.reshape(-1, hidden_size)
pp.pprint(x_data)

softmax_w = np.arange(15, dtype=np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data, softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes) # batch, seq, class
pp.pprint(outputs)



# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())




# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

prediction3 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
prediction4 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction2, y_data, weights)
sequence_loss3 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)
sequence_loss4 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval(),
      "Loss3: ", sequence_loss3.eval(),
      "Loss4: ", sequence_loss4.eval())

