#TensorBoard

#tensorbard 는 deep하고 wide하게 만들 때, 그 진행과정을 만드는 유용한 방법임. (그래프를 그림)
#step에 따라서 loss가 어떻게 변화하는지를 볼 수 있음.
# 이전에는 단순히 print로 본 것에서 upgrade 한 것임.

#아래의 5개의 step을 따라간다면, 멋진 그래프를 그릴 수 있음.

############ 요약 ############

#### 1 From TF graph, decide which tensors you want to log
#### w2_hist = tf.summary.histogram('weights2',w2)
#### cost_summ = tf.summary.scalar('cost',cost)

#### 2. Merge all summaries
#### summary = tf.summary.merge_all()

#### 3. Create writer and add graph
#### writer = tf.summary.FileWriter('./logs') #full path 입력해보기. // 어느위치에 경로를 적을껀지
#### writer.add_graph(sess.graph) #같이 쓰면 됨.

#### 4. Run summary merge and add_summary #summary자체도 tense이기 때문에 실행시켜야함.
#### s, _ = sess.run([summary , optimizer],feed_dict=feed_dict)
#### writer.add_summary(s , global_step=global_step) #실제로 파일에 기록하는 함수.

#### 5. Launch TensorBoard #터미널 가서, 정해진 디렉토리를 '=' 오른쪽에 넣기
#### tensorboard --logdir=./logs     #"--logdir=./logs" 띄어쓰기 없이 하기

#############################################################################################



############ 순서 시작 ############

# 1. 열심히 코드를 실행한다. (import부터 for문 다 돌리는거까지)
# 2. 터미널에 [[[ tensorboard --logdir=./logs ]]] 를 실행한다.
#### 이 의미는 , [[[ C:\Users\82104\PycharmProjects\untitled\logs]]] 를 실행하는것. ###
# 혹은, [[[ tensorboard --logdir=path/to/log-directory]]] 를 실행한다.
# 이때, writer = tf.summary.FileWriter("./logs/xor_logs") 와 같이 되어있을 것이다.
# 참고 사이트 :  https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/how_tos/summaries_and_tensorboard/

# 3. 인터넷에 가서 [[[ localhost:6006 ]]] 를 입력하고 확인한다.
# 4. 터미널을 종료하려면 [[[ ctrl+c ]]]를 누른다.

# 5. 만약 여러개를 실행하고 싶으면,
# 5-1 )
# 코드 중 [[[ writer = tf.summary.FileWriter("./logs/xor_logs") ]]] 로 바꾼다.
# 또한 다르게 돌릴 목적에 맞게 코드를 수정한다. (ex:  learning_rate = 0.1)
# 터미널에 [[[  tensorboard --logdir=./logs/xor_logs  ]]] 하고 데이터 돌리고 터미널 종료.
# 5-2)
# 코드 중 [[[ writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")  ]]] 로 바꾼다.
# 또한 다르게 돌릴 목적에 맞게 코드를 수정한다. (ex:  learning_rate = 0.001)
# 터미널에 [[[  tensorboard --logdir=./logs/xor_logs_r0_01 ]]] 하고 데이터 돌리고 터미널 종료.
# 5-3)
# 그리고 최종적으로 [[[ tensorboard --logdir=./logs  ]]] 를 실행
# 이러면 동시에 볼 수 있게 된다.

#6. 만약 서버를 할당받고 싶으면,
# 터미널에 [[[ ssh -L 7007:121.0.0.0:6006 donggyu@server.com ]]] 이라 하면,
# 6006번에 해당하는게 7007번으로 donggyu이름을 지닌채 할당받음.
# 이후 , 인터넷 주소창에 [[[ localhost:7007  ]]] 을 입력한다.

############ 순서 끝 ############

#Tensorboard Tip)
# 1. Scalars
# smoothing : 이동평균을 취해서 부드럽게 하라는것. 0값을 넣으면, 원데이터 그대로를보여줌.
# 나머지는 직관적으로 볼 수 있음.

#2. Graphs
#우리가 어떻게 코딩을 했고, 어떤 알고리즘을 갖는지 시각적으로 표현하는 파트.
#with문을 쓰고, name_scope를 쓰는 이유는 이 파트에서 깔끔하게 보기 위함.
#그래프들을 더블클릭해서 세부 계산 과정을 볼 수 있음. 확인 해보기.

#3. Histograms
# y축은 step, x축은 벡터들의 원소가 가지는 각 값에 대한 분포임.
# 즉, step 이 경과함에 따라 어떻게 값이 변화하는지를 볼 수 있음.

###############################
##### Tensorboard
# Lab 9 XOR
import tensorflow as tf
import numpy as np


tf.set_random_seed(777)  # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name="x") #아래에서도 마찬가지지만, 이름을 정하자.
Y = tf.placeholder(tf.float32, [None, 1], name="y")

#Add scope for better graph hierarchy
#그래프를 그릴때, layer1과 layer2를 보기 좋게 만들 수 있음.
#아래처럼 하는걸 습관화하자. (tf.name_scope는 tensorboard에서 그래프 구성을 깔끔하게 볼 수 있도록 도와줌.)

with tf.name_scope("Layer1"): #Layer1은 이름정한것.
    W1 = tf.Variable(tf.random_normal([2, 2]), name="weight_1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias_1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    tf.summary.histogram("W1", W1) #tf.summary 매서드를 잊지말기.
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([2, 1]), name="weight_2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias_2")
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Hypothesis", hypothesis)

# cost/loss function
with tf.name_scope("Cost"): #cost와 train 또한 그래프로 만들어 버린다. 습관화.
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.summary.scalar("Cost", cost) #cost는 scalar이기에, histogram 문법을 쓰지 않는다.

with tf.name_scope("Train"):
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
#AdamOptimizer 란,
# Momentum(모멘텀) + RMSprop(알엠에스프롭) 기법이 혼합된 최적화기법
# (관성 효과 + 지수이동평균 의 혼합이다.) 이런게 있다 하고 넘어가기.
#다음의 사이트를 참고하면 Optimizer들에 대한 이야기를 자세히 살필 수 있다.
# https://twinw.tistory.com/247

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
tf.summary.scalar("accuracy", accuracy)  #accuracy 또한 scalar이기에, histogram 문법을 쓰지 않는다.

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs_r0_01 였음.
    merged_summary = tf.summary.merge_all()  #요약 할것을 다합쳐버리고,
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01") # 이 경로에 정보를 넣는다.
    # 이의미는 , C:\Users\82104\PycharmProjects\untitled\logs\xor_logs_r0_01 를 실행하는것.
    writer.add_graph(sess.graph)  # Show the graph  & Add graph in the tensorboard

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, summary, cost_val = sess.run(
            [train, merged_summary, cost], feed_dict={X: x_data, Y: y_data}  #train 시키면서 요약된 값들도 얻어내고,
        )
        writer.add_summary(summary, global_step=step) #요약본을 step이 진행될수록 계속 합쳐버리면서 누적시킨다. 이때 writer변수는
                                                      #경로에 해당하는 변수이고, 데이터가 저장되는 공간 (위에서 정의함.)
        #add_summary(summary,step) 와 add_graph(sess.graph)를 잊지 말기.

        if step % 1000 == 0:
            print(step, cost_val)

    # Accuracy report
    h, p, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}
    )

    print(f"\nHypothesis:\n{h} \nPredicted:\n{p} \nAccuracy:\n{a}")


##### 과제 해보기. mnist