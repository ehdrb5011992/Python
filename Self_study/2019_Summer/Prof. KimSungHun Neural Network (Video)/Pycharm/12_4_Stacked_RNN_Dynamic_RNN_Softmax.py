from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
#문장이 길어지면, 위처럼 ""로 연결해서 쓴다는 사실을 기억하자.
#len(sentence) 문장길이는 180
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}
#각 값들의 인덱스가 value, 문자들이 key임.

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
#그냥 10개 정도로 sequence_length를 정한다. 단어를 넣어서 학습할꺼임.
learning_rate = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length): #단어를 학습시키는 전처리 과정.
    x_str = sentence[i:i + sequence_length] #input부분. 10개이다.
    y_str = sentence[i + 1: i + sequence_length + 1] #output 부분. 역시나 10개이다.
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    #char_dic은 dictionary고, char_dic[c]를 하면, key==c에 해당하는 value를 출력한다.
    #즉, c와 x_str은 문자이고 char_dic[c]는 숫자값이 출력.
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)
# 일반적인 파이썬 코드를 array에 쌓는다.
# dataX와 dataY는 10개의 input,output들을 지닌 2차원 숫자 데이터 배열이다.


batch_size = len(dataX)
# 이 개념을 사용해야한다.
# 철자 하나하나 부여하면, sequence_length가 길어지고, 답도없이 계산량이 많아지기 때문.
# 169(?)가 될 것이다. <--- 170인데요
# 이중 리스트의 길이는, 안의 리스트들의 개수임.

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape


# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell
# 쉽게 보려고, 이렇게 함수로 만든 후, lstm_cell() = cell 이라고 말함.
# 이렇게 쌓을 경우 *랑 차이점은, 각 층마다 다른 형태의 cell을 쌓을 수 있다는 것.
# * 보단, 위처럼 쌓기.

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
#for _ in range(2) 는 for i in range(2) 와 같은거임. 그냥 2번 실행해라.

#이전에서는 다음처럼 표기했음.
#cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
#이 함수를 하고, * 2개 이러면 2개 쌓아버리는거임.

#위 코드를 아래처럼도 물론 사용할 수 있음.
#multi_cells = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)


# outputs: unfolding size x hidden size, state = hidden size
#sequence 데이터를 처리할 수 있는 RNN
#그러나, 실전에서 sequence가 정해지지 않는 경우가 있다.
#예를들면, 번역할 때 중간중간에 단어가 없어도 번역이 될 수 있어야함.
#이런걸 어떻게 처리할까?
#이는 비어있는 문자에 padding을 넣어주면 됨.

#이를 Different sequence length 로 해결.
#각 배치에 sequence의 길이를 정의하게 됨.
#즉, 단어 마다 각 배열의 길이를 먼저 측정해서 sequence를 출력함.
#그리고 확실하게 없는 데이터에 대해서는 0으로 만들어버리고 정보를 없앰
#이게 dynamic_rnn임.
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# FC layer -> 잘 되니까 이렇게 사용.
X_for_fc = tf.reshape(outputs, [-1, hidden_size]) #기계적으로 이렇게 해준다.
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# x_for_softmax = tf.reshape(outputs,[-1,hidden_size])
# softmax_w = tf.get_variable('softmax_w',[hidden_size,num_classes])
# softmax_b = tf.get_variable('softmax_b',[num_classes])
# outputs = tf.matmul(x_for_softmax,softmax_w)+softmax_b
# outputs = tf.reshape(outputs,[batch_size,seq_length,num_classes])


# reshape out for sequence_loss #데이터가 섞이지 않고, 형태를 바꿀 수 있다.
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
#위의 output을 넣는것이 맞는 답임. 그전에 actiavation ftn이 거치지 않는 것을 넣어야함.

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
#위의 함수 안에 softmax함수가 있음.

mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#학습단계
for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        # results는 [batch_size, sequence_length, num_classes] 차원 데이터.
        # (170,10,25) 임. 10단어씩 쪼개면서 학습을 하는데, 그게 170번 있는것.
        # 즉, j는 170번 돌아감.
        # 전체 문장길이는 180개의 철자. 25는 문자들의 one_hot 값.
        # results는 숫자 값들이 들어가있다.
        index = np.argmax(result, axis=1)
        # result는 10*25 행렬임. 이중에 가장 안쪽의 값중 큰 값을 반환함.
        print(i, j, ''.join([char_set[t] for t in index]), l)
        # 단어들이 어떻게 학습되고 있는지 확인.

# Let's print the last char of each result to check it works
# 이렇게 위에서 학습해서 얻은 outputs을(모형을) dataX에 넣고 실행시켜보자.
results = sess.run(outputs, feed_dict={X: dataX})
#results.shape 는 (170,10,25)이다.

for j, result in enumerate(results): #각 배치의 출력을 할 수 있다. (170번)
    index = np.argmax(result, axis=1) #result는 행렬이고, 이에 따라 index는 벡터가됨.
    if j is 0:  # print all for the first result to make a sentence
        #처음의 경우만 문장을 형성하고 10개짜리 단어를 한번에 만듦.

        print(''.join([char_set[t] for t in index]), end='')
        # 끝에 무엇을 쓸 지에 대한 옵션으로 end를 쓰며, ""을 하는 이유는 아래에 설명.
        # 무엇을 사용할것지에 대한 내용은 아래의 사이트를 참고.
        # https://it-coco.tistory.com/entry/%EC%B6%9C%EB%A0%A5-print-sep-end
    else: #이후에는 1개씩 단어를 추가함. 그렇게 169번을 추가. 그러므로 앞의 1글자가 짤린다.
        print(char_set[index[-1]], end='')
        # end=""를 했기 때문에 지속적으로 print문을 사용했을 때 이어서 출력되는 효과를 보임.
        # -1인 이유는, 앞의 9글자는 계속 겹치기 때문. 뒤에만 새롭게 늘어나는것이다.

#모든 학습이 끝나면 굉장히 원래의 문장과 유사하게 나온다. - 학습완료.
#RNN을 통해서 코드를 직접 작성할 수 있음.