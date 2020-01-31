#### Class ####

# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 2 #기니까... 2로 수정했음.
batch_size = 100


#파이썬의 클래스로 좀더 효과적으로 관리할 수 있다.
class Model:
    # m1이 self임.
    def __init__(self, sess, name): #초기값 설정. self변수는 내가 앞으로 설정할 변수의 이름을 받는 부분이므로, 이 함수의 실질 변수는 2개인 셈이다.
        self.sess = sess # self.sess라는 변수에는 sess를 입력
        self.name = name # self.name이라는 변수에는 name을 입력
        self._build_net() # 아래의 _build_net()이라는 함수를 실행한다.
        #이렇게 변수들을 저장하면 pypy 라는 class를 정의할 시 , pypy.sess ,  pypy.name 라고 자동으로 변수들이 저장된다.
        #session을 넘겨주면 편하다.

    def _build_net(self): #전체적으로 변수 앞에 self가 들어갔다.
        with tf.variable_scope(self.name): #self.name = name 이었으므로, 실제 우리가 넣은 name에 관련되어 scope가 저장됨.
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1]) # (n,p,q,depth)
            self.Y = tf.placeholder(tf.float32, [None, 10]) #결과적으로 분류할 Y

            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) #3*3*1 필터 32개 (p,q,depth,filters)
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') #same이므로, 아직까지 L1은 28*28 (stride=1이기에)
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME') #최대만 뽑아내고,
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob) #그 중 dropout으로 뽑아냄.
            #계속해서 1개의 데이터 내에서 보고있는 상황임.
            '''
            Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
            Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
            '''

            # L2 ImgIn shape=(?, 14, 14, 32) <- Activation Map을 이야기 함.
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
            Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
            '''

            # L3 ImgIn shape=(?, 7, 7, 64)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)) #필터는 3*3 , L2 후에 pooling 한 Activation Map은 7*7 (즉 옆의 변수의 L2는 7*7)
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128) #pooling 하면 이렇게 뜸. // (9-3)/2 + 1 = 4
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') #stride = 1  // 그래서 L3도 7*7 (same 이기 때문)
            # (F-1)/2 = zero padding의 수 이므로, 0으로 둘러 쌓은 벽이 1겹 생성됨. // => (7 - 3 + 2 ) / 1 + 1 = 7
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME') # =>   (9-3)/2 + 1 = 4
            #https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
            #엄격한 정의는 위를 참고하고, 내가 이해한 것과 기본은 같다.

            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4]) #
            '''
            Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
            Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
            Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
            Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
            Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
            '''

            # L4 FC 4x4x128 inputs -> 625 outputs
            #이 부분은 FC으로 펼쳐서 작업하는거.
            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                                 initializer=tf.contrib.layers.xavier_initializer()) #여기서 초기값을 얻을때는 get_variable 함수를 통해, 자비에 초기값을 정함.
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
            '''
            Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
            Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
            '''

            # L5 Final FC 625 inputs -> 10 outputs
            W5 = tf.get_variable("W5", shape=[625, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L4, W5) + b5 #마지막은 그냥 이 자체로 쓰자. (어차피 softmax 안써도 1-1 이기 때문에 계산량 줄임.)

            '''
            Tensor("add_1:0", shape=(?, 10), dtype=float32)
            '''

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( #그래도 cost함수는 softmax의 cross_entropy를 사용.
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer( #아담사용.
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal( tf.argmax(self.logits, 1), tf.argmax(self.Y, 1) )
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #prediction과 정확도까지 끝.

    ###### 여기까지가 _build_net 함수였다. #########


    def predict(self, x_test, keep_prop=1.0): #예측값들 출력. logits만으로 끝냈기에, 확률을 출력해주는 것은 아니다.
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})
        #위에서 self.X , self.keep_prob를 placeholder로 변수 저장했음.
        # m1이 self임.

    def get_accuracy(self, x_test, y_test, keep_prop=1.0): #정확도 출력. self.accuracy를 실행함.
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7): #학습실행. cost를 출력한다. optimizer는 아무것도 출력하지 않음.
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
#이렇게 만들면 쉽게 간단하게 학습을 시킬 수 있다.

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer()) #이렇게 초기설정.

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0 #방생성.
    total_batch = int(mnist.train.num_examples / batch_size) #range로 받기 위해 int로 설정함을 잊지말자.

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys) #우리가 위에서 class에서 정의한 도움함수.
                                            #train은 cost와 optimizer함수가 같이 있었고, 우리는 cost만 출력해서 볼거기 때문에 나머지를 _ 로 입력.
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

#우리가 앞으로 클래스를 사용하면 더 편리하게 코딩을 할 수 있다.

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))


######################################

#### Layers ####

# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 2 #기니까 2으로 수정.
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool) #self.training을 T,F로 받을것인지에 대해 placeholder을 만듦! 보고 넘어가기.

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], # strdies = (1,1) 이 초기값으로 들어가있는 상태임.
                                     padding="SAME", activation=tf.nn.relu)
            #아래를 위처럼 간략하게 나타낼 수 있다. 숫자를 보기가 편해짐. 이렇게하기.
            #####################################################################################
            # W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))                    #
            # L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')                #
            # L1 = tf.nn.relu(L1)                                                               #
            #####################################################################################


            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.3, training=self.training )
            #30%를 가지고, 70%를 버린다는 뜻임.... 이러면 일반적으로 학습효과가 그닥이라고 함
            #dropout 은 training이라는 옵션에 T,F를 줘서 에러를 방지 할 수 있음! self.training은 위에서 정의한 변수.
            #####################################################################################
            # L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #
            # L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)  # 그 중 dropout으로 뽑아냄.      #
            #####################################################################################

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.3, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4]) #이런걸 계산하기 위해서 (N-F +2P )/S +1 을 알아야함.
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu) #출력을 625로. 내마음대로 정함.
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training) #dropout비율은 50%로.

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10) #궁국적으로 10개

        ########################## 코드가 훨씬 더 깔끔해짐 #########################

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ##### 여기는 위와 같음.

    def predict(self, x_test, training=False): #이렇게 training을 boolean 타입으로 쓰고 받아도 문제가 없게 됨.
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

################ 걍 코딩을 외우자. - 언젠가는 ################

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))


##########################################################################

#### 앙상블 ####

# Lab 11 MNIST and Deep learning CNN
# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 2 #20인걸 편의상 1만 하자.
batch_size = 100

###################### 아래 class Model 은 바로 위에서 한 것과 같다 ###################
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784]) #데이터를 받을 그릇.

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10]) #역시나 데이터를 받을 그릇.

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.3, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10) #activation에 아무것도 안줬으므로, 그대로 출력하는거임.
                                                                     #tf.nn.sigmoid , tf.nn.softmax 등이 올수 있음.
                                                                     # (데이터 * 10) 차원임.

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

#####################################################################################

# initialize
sess = tf.Session()

models = [] #그릇을 만들어놓고
num_models = 2 #2개, 3개, 10개 마음대로 앙상블 할 개수를 선택.
for m in range(num_models):
    models.append(Model(sess, "model" + str(m))) #모델0, 모델1 을 각각의 models 리스트에 하나씩 저장. 이때 초기값이 다르므로, 다른 모델이 형성됨.
#이러면 클래스 instance가 생성됨.

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):  #이건 전체 학습시킬 횟수. 1번 하기로 했다.
    avg_cost_list = np.zeros(len(models)) # 0을 원소로 하는 2차원 벡터가 생성됨. 여기다가 cost들을 계산해낼꺼임.
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch): # 데이터는 일정하게 100개씩 뽑아올꺼임.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models): #뽑아온 100개의 데이터를 각 모델들에 적용하고, 각각의 cost를 출력해냄.
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
        #m을 꺼내와서 학습을 시킴.

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10]) #행은 test 할 크기(데이터=1만), 열은 0~9에 해당하는 변수.
#즉, 데이터*10 차원의 행렬에 0을 원소로하는 공간을 생성.

#공간을 만들어버린다.
for m_idx, m in enumerate(models): # 0 , 0번째 모델 / 1, 1번째 모델 // 이렇게 돌아감.
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels)) #각 모델의 정확도를 일단 출력하고,
    p = m.predict(mnist.test.images) #각 모델의 예측값을 출력한 뒤 p변수에 저장한다.
    # 이때 , mnist.test.images는 10000*784 짜리 테스트 데이터들이고,
    # m.predict를 시행하면 각 모델에 해당하는 predict를 시행하게 된다. 이때 predict는 logits을 계산해줌을 명심하자.
    # mnist.test.images 는 test 변수로 새롭게 넣을 X데이터에 해당하고, 만들어진 모델 m 에 따라서 예측치를 출력한다.
    # 이때, 출력하는 예측치의 차원은 데이터*10 차원 행렬이다.

    predictions += p # 10000*10 행렬을 누적해서 통째로 더하는 과정임.

#각각의 모델들 학습. (C_1,C_2, ... , C_m) // 우리는 2개의 모델이므로, C_1 , C_2
#각 모델들 예측 시켜버리고, 각 케이스별로 , label별로 확률 값을 더해서 가장 큰 label의
#수치를 선택해버리면 됨.(간단한 방법)
#사실 엄밀히는 확률이 계산되는게 아닌, 로짓이 계산됨. 우리는 위에서 softmax를 사용하지 않았기 때문.
#그렇다해도, 큰 수치를 선택하면 되는 사실엔 변함이 없음.
#그게 위의 코딩.

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
#좋은 성능을 관측 할 수 있다.

#########################################################################

#### Batch Size ####

# batch size에 따른 모델의 성능비교.
# MNIST가 아닐때, batch size만큼 뽑아올때 어떻게 할것인지에 대한 idea도
# 여기서 얻을 수 있다.

# Lab 10 MNIST and Deep learning CNN

import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 2
batch_size = 100

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128 * 4 * 4])
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning stared. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


########################### 이 위는 전부 같다. 아래를 살펴보자. ############################

def evaluate(X_sample, y_sample, batch_size=512):
    """Run a minibatch accuracy op"""

    N = X_sample.shape[0] # N 은 X_sample의 행의 수.
    correct_sample = 0 #초기값 설정.

    for i in range(0, N, batch_size): #이렇게 주게되면, i를 batch_size에 해당하는 것부터 띄엄띄엄 시작해서 주게된다. 꼭 기억하기.
        X_batch = X_sample[i: i + batch_size] #batch_size만큼 행을 뽑음. //  X_sample[i: i + batch_size , :] 와 같음.
        y_batch = y_sample[i: i + batch_size]
        N_batch = X_batch.shape[0] #행의 수. = 배치의 크기. batch_size라 봐도 무방하다.

        feed = {
            X: X_batch,
            Y: y_batch,
            keep_prob: 1
        }

        correct_sample += sess.run(accuracy, feed_dict=feed) * N_batch #배치 크기만큼 곱해줌. (비중을 곱하는 것.)

    return correct_sample / N

print("\nAccuracy Evaluates")
print("-------------------------------")
print('Train Accuracy:', evaluate(mnist.train.images, mnist.train.labels))
print('Test Accuracy:', evaluate(mnist.test.images, mnist.test.labels))


# Get one and predict
print("\nGet one and predict")
print("-------------------------------")
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r + 1], keep_prob: 1}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()


