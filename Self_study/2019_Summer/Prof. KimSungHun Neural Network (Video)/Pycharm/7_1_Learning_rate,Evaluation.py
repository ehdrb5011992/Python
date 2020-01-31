# Lab 7 Learning rate and Evaluation

# 1. learning rate
# 우리는 gradient descent를 할때 alpha(=learning rate)를 임의로 정의했음.
# 이때, 이 learning rate를 잘주는게 중요함. step이 너무크다면,
# 제대로 수렴을 못할 수 있음. (바깥으로 튕겨저 나갈 수 있음.)
# 이를 overshooting이라고 함.(cost가 크게 나타나는 경우 - 내가경험한 그것)
#
# 우리가 굉장히 작은 learning rate(step)을 주게 된다면, 최저 점이 아님에도
# 불구하고 stop하게 될 수 있음. -> cost함수를 보고 얼마나 변하는지를 보는게 중요.
#
# learning rate를 설정하는 답은 없다. 단순히 lambda값은
# 발산 -> 좀 더 작게 // 수렴에 의문이든다? -> 좀 더 크게
# 로 한다.


# 더불어
# 우리는 그동안 데이터를 통해 ML(머신러닝) 모델을 학습시킴.
# 이때 이를 평가하는 법에 대해 알아보자.
#
# 데이터가 얻어지면 30%정도를 짜르고,
# 30% -> training set (얘로만 모델을 학습시킨다)
# 70% -> test set으로 둠. (얘는 절대로 사용해서는 안됨.)
#
# 후에 단 한번의 기회를 가지고 70%데이터들이 얼마나 모델에 잘맞는지 봄.
# (예측치와 참값을 비교해가면서)
# --아직은 반드시 이렇게 해야함--
# ----------------------------------------------------------------
# 혹은 좀 더 구체적으로 나누면,
# Training/ Validation/ Testing 이렇게 3개로 나눔.
# Validation은 모의시험임. 그리고 얘를 training에서 얻어진 모형에서
# 조율모수들(alpha, lambda 등) 을 설정하는 해답을 갖게됨.
# 이후에 Testing을 함.

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
####이게 train // 앞으로는 반드시 데이터셋을 나눈다.


x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1], #one hot
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=10).minimize(cost)
#learning_rate 를 1e-10로 줘도, 결과가 매우 이상하게 나옴. <- learning_rate조절 잘하자!!
#이 경우 올바른 learning_rate는 0.1 정도.

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#################여기까지 전부 똑같다###############
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        #학습은 x_data - train으로,
        print(step, cost_val, W_val)

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test})) #예측은 test / 예측은 feed_dict input이 1개
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test})) #정확도도 test / 정확도는 feed_dict input이 2개

    #accuracy , prediction 는 학습이 되고 고정이 됨. (모형적용)


#######################################################

# Non_normalized inputs

# 2. data 선처리
# x1 = (1,2,3,4)
# x2 = (10000, -9000, 8000,-2000) 이렇게 매우 큰 차이가나게 된다면,
# 2차원에서만 해도 매우 찌그러진 원 형태의 등고선이 그려짐. 그리고 이는 다차원에서도 마찬가지.
#
# 이런 경우 우리가 learning rate 어떻게 주게 될지 난감하게됨
# ->> normalize  / zero-centered data 등의 행위를 해서 조절한다.
# normalize는 내가아는 statistical standardization 외에 여러개 있음.

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility


#3번째 열들은 굉장히 큰 값을 지닌 행렬이다.
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1] #마지막 열 빼고 나머지
y_data = xy[:, [-1]] #마지막 열

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
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

for step in range(101):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    print(step, "Cos: ", cost_val, "\nPrediction:\n", hy_val)


###########################################################

# Normalized inputs

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

#이런 함수를 정의함.
#이걸 사용하게 되면, 가장 큰걸 1, 가장작은걸 0으로 기준하고
#나머지는 비율에 맞춰 표준화를 시키게 됨. 유니폼하게 선형으로 연산됨.

# 혹은 다양한 방식으로 데이터를 표준화 시킬 수 있음.
# https://medium.com/@swethalakshmanan14/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
# 위의 사이트를 참고할 것.
def min_max_scaler(data):
    numerator = data - np.min(data, 0) #각 열기준 최솟값을 데이터 원소에서 빼는것.
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7) # 최소한의 분모를 1e-7로 고정.

#행렬을 이쁘게 표현하는 법. 숙지하기.
xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

# very important. It does not work without it.
xy = min_max_scaler(xy) #이러면 원소별로 적용이된다.

print(xy)
#이후 동일.

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, cost_val, hy_val = sess.run(
            [train, cost, hypothesis], feed_dict={X: x_data, Y: y_data}
        )
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

##학습이 잘 이루어진다.

#그 외에...

# 3. 머신러닝의 가장 큰 문제인 overfitting
# 머신리닝이 학습을 통해 만들어질때, 학습데이터에 너무 잘맞는 모델을
# 만들어 낼 수 있음.
#
# -> 해결방법
# : 트레이닝 데이터가 많으면 많을수록.
# : 우리가 가지고 있는 features 수를 지우기
# : Regularization(일반화) 시키기
#
# 좀더 자세히, Regularization 를 하라는 뜻은,
# -너무 많은 weight를 주지 말자.- 라는 뜻.
# 우리가 오버피팅 한다는 의미는 weight을 주고 직선을 더욱더 곡선으로 만드는
# 행위를 이야기함.
#
# 이런 regularization의 예로, Lasso, Ridge 등이 있음. (cost함수 뒤에 lambda * sum w^2을 더함)
# 이때 lambda를 regularization strength라고 함. 클수록 regularization을
# 중요하게 생각한다는 뜻. (조율모수)



