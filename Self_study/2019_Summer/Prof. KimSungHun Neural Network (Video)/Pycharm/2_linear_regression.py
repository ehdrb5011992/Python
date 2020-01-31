import tensorflow as tf

#1. Build Graph Using TF operations
tf.set_random_seed(777)  # for reprducibilty

#X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3] #굉장히 간단한 트레이닝데이터.
# H = Wx + b 에서 W와 b를 찾아볼것.

#Variable
# Try to find value for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let's TensorFlow figure it out
#텐서플로우가 자체적으로 생성하는 변수. (혹은 trainable한 variable임. 학습하는 과정에서 변경시킴.)
W = tf.Variable(tf.random_normal([1]), name='weight')
#[1] 은 rank가 1인 1차원 array shape을 줌.
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our Model
# Out hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#cost(W,b) = 1/m sum(제곱오차)
#reduce_mean 은 그냥 평균내주는것.

#minimize
#GradientDescent를 써서 최적화함.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)



#Run / update Graph and get results
#Prepare session
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer()) #이걸 반드시 실행해줘야함. 초기화하는행위.
#r에서 rm(list=ls())처럼 외우기. sess.run(tf.global_variables_initializer())
#Fit in line
for step in range(2001):
    sess.run(train)
    if step % 200 == 0: #트레인을 훑어보면서. 확인.
        print(step, sess.run(cost), sess.run(W), sess.run(b))

#train 아래에 cost아래에 hypo 아래에 W와 b변수가 있음. 즉, train을 실행시키는 것으로 학습이 일어나서 W,b가 얻어짐.
#이런 행위는 코드를 짜는 일련의 행위임. 그냥 내가 지금껏 해오던 행위.
#W,b를 가지고 hypo함수 만들고, 그걸로 cost함수 만들어서 train을 시킨것임.
#W=1 , b=0이 정답이라고 할 수 있겠고, 그 결과를 볼 수 있음.

#######################################################################

#Placeholders
tf.set_random_seed(777)  # for reprducibilty

#Variable
# Try to find value for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let's TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#X and Y data
#X와 Y를 직접 주는것이 아니라, placeholder이름으로 주는것.
#### Now we can use X and Y in place of x_data and y_data
#### placeholders for a tensor that will be always fed using feed_dict
#### See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None]) #1차원 array , None은 무수히 많이 줄수있다는 뜻.
Y = tf.placeholder(tf.float32, shape=[None])


#Our Model
# Out hypothesis XW+b
hypothesis = X * W + b
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Prepare session
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())


#Fit the line
#placeholder이기 때문에 feed_dict를 통해서 넘겨줄 수 있게됨
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], #리스트에 넣어서 한꺼번에 실행할 수 있음.
                                      # train은 필요가 없음.
                 feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 200 == 0:
        print(step, cost_val, W_val, b_val)
#즉, 리니어모델을 먼저 만들어 놓고 뒤늦게 데이터를 던져주고 학습시키는 행위임.

#Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))


#Fit the line with new training data
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 200 == 0:
        print(step, cost_val, W_val, b_val)
# W = 1 , b= 1.1 로 예상

#Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]})) #5를 넣었을때,
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]})) #2개를 넣어줄수도 있음.




# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Model parameters
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * W + b

# cost/loss function
cost = tf.reduce_sum(tf.square(hypothesis - y))  # sum of the squares

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    W_val, b_val, cost_val = sess.run([W, b, cost], feed_dict={x: x_train, y: y_train})
    print(f"W: {W_val} b: {b_val} cost: {cost_val}")

