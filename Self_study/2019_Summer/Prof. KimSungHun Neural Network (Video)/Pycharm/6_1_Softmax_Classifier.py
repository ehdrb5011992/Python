# Lab 6 Softmax Classifier
#여러개의 클래스가 있을때, 그것을 예측하는 multi classification, 그 중 자주 사용하는
#soft max에 대해 알아보자.
#로지스틱 함수를 찾는다는건, 무리를 구분하는 선을 찾는것과 같은 의미.
#이는 soft max에서도 마찬가지이다.
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1], #2
          [0, 0, 1], #2
          [0, 0, 1], #2
          [0, 1, 0], #1
          [0, 1, 0], #1
          [0, 1, 0], #1
          [1, 0, 0], #0
          [1, 0, 0]] #0
#이렇게 encoding을 해줘야한다. RNN과 비교해서 나중에 알아놓기.
#one - hot encoding : 하나만 핫하게 하는 인코딩 -> 핫하다는건 1이다라는 의미고,
#위에서 사용한 y_data가 one hot encoding임.
#마치 선형모형에서 design matrix 처럼 바뀜.

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])

#Y의 3도 사실 아래의 nb_classes임.
#y열의 개수는 label의 개수임. (class의 개수임.)
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
#nb_classes = 3 이었다.
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#W와 b는 구분을 제대로 해주는 모수라고 쉽게 생각하자.
# ->> Y = XW + b
# ->> (n x 3 ) = (n x 4 ) (4 x 3) + (n x 3) / 차원
# ->> 1개의 데이터에 대해서, (1 x 3) = (1 x 4 ) (4 x 3 ) + (1 x 3) / 차원


# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# 위는 softmax의 정의.
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
#tf.sigmoid 대신 tf.nn.softmax , 얘는 확률값. softmax모형임.
#즉, 확률로 점수를 주는 행위가 softmax덕에 가능해짐.
#다시한번 잊지말기! softmax는 매번 얻어지는 모형을 통한 class의 분류 확률값들을 제공함.
# // hypothesis 는 ( 데이터의 수 x  3(class) )인 행렬이고, 각 값들은
#예측 확률로 구성되어있음. //
#W를 계산하는 과정에서, gradient descent 알고리즘이 들어가게 되고,
#이때는 Y데이터 값을 사용하기에 예측 확률은 학습을 하면할수록 갱신된다.


# Cross entropy cost/loss
cost = tf.reduce_mean(  -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)   )
#여기서 Y의 존재의미가 담김. cost에서 Y를 쓴다. <--- 분류의 판단정보가 여기서들어감
#행렬의 *는 요소별곱임.
#reduce_sum의 옵션 axis = 1 은 행단위로 더하라는 뜻. (행렬에서)
#정확한 의미는 차원축 하나를 제거한 뒤 나머지를 더하라 라는 의미이다. 관심있으면 찾아보기.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#또한 그 행위는 gradient descent의 행위를 반복함으로써 학습시키고,
#점점 더 정밀해짐.

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]}) #행렬로 주는것을 잊지말자.
    # argument에 [hypothesis]보단 hypothesis가 출력결과를 보기 깔끔함.
    #a는 각 클래스별 확률값을 지닌 벡터가 출력이 된다.
    #hypothesis는 soft max까지 적용된 결과값이었음.
    print(a, sess.run(tf.argmax(a, 1)))
    #argmax의 1은 axis=1옵션 - 행기준비교(행중에서)
    #tf.argmax 는 어느 argument가 최대인지 index를 출력함. <- 리스트 형태로 출력됨.

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                              [1, 3, 4, 3],
                                              [1, 1, 0, 1]]}) #데이터를 행렬로 줘보자.
    print(all, "\n", sess.run(tf.argmax(all, 1))) #역시 axis=1로 해야 각 행에서 최댓값 출력.

