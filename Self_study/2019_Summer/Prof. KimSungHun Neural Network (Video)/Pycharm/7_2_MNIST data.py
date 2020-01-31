# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# MNIST 데이터 - 우체국에서 우편번호 숫자 쓸때 그 인식과 관련된 데이터

# Online learning이라는 학습이 있음.
# 100만개의 training set이 있다고 한다면, 한번에 시행하는 것이 아닌
# 10만개씩 잘라서 학습을 시킨다.
# 이때, 모형만은 남아서 추가로 계속 학습이 되어야함.
#
# 굉장히 좋은 아이디어이며,  추가로 들어올 데이터에 대한 학습의 여지도 남겨놓기 때문.
# MINIST 데이터는 굉장히 유명한 숫자 손글씨 데이터.
# 이런 데이터도 데이터가 나눠져있음.
# ----------------------------------------------------------------
# 이후 정확도에 대한 관심사도 빼놓을 수 없음.(최종목표)
# Y값과 모델이 예측한 값의 비교를 하고, 얼마나 맞추는지를 이야기함.
# 최근 이미지 정확도는 적어도 95%를 넘기고 있다. 상당히 정확한 편.

tf.set_random_seed(777)  # for reproducibility
from tensorflow.examples.tutorials.mnist import input_data #그 많은 데이터들 중 mnist input_data만 불러옴.

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
# mnist 데이터셋 정리1 : https://pythonkim.tistory.com/46
# 데이터 파일이 궁금하다면? : https://m.blog.naver.com/PostView.nhn?blogId=acwboy&logNo=220584307823&proxyReferer=https%3A%2F%2Fwww.google.com%2F


#input_data만 함수로 받아왔기에, 아래와 같이 씀. 만약 import로 tensorflow.examples.tutorials.mnist 를 했으면,
#tensorflow.examples.tutorials.mnist.input_Data.read_data_sets로 매우 길어짐... 읽기불편 // 이게 from ~ import를 쓰는 이유.

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#데이터가 Mnist가 없으면 자동으로 설치하고, mnist에 자동으로 변수저장이 된다.
# "MNIST_data/" 의 경로는 바탕화면이다.(?) <- 내가 경로를 바탕화면에 저장해서 그렇게 생긴걸듯..
# 다시말해, MNIST_data폴더가 생성되고, mnist변수에 mnist 데이터를 one_hot으로 불러오라는 명령어.

#y값을 one_hot으로 처리해서 불러오는 옵션. 반드시 True로 돌릴것.
#물론 데이터가 one-hot 방식으로 넘어오면 쉽게 처리할 수 있다.
#mnist를 출력해보면, 데이터셋이 train, validation, test로 구성되어있는거 확인가능.

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
# 28*28 = 784 픽셀로 이루어짐. 즉, 784개의 x변수들이 있음. 칼라가있으면 784*3이 됨.
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
# Y는 0~9로 출력이 됨. 즉, 10개의 변수들.
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes])) #y의 사이즈

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#tf 1.14.0 에서 부터 argmax 를 사용하도록 권장. (arg_max보다)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#tf.cast에 tf.float32이 타입옵션으로 들어감을 잊지말기.



# parameters
num_epochs = 15 #15번 학습함. (많을수록 좋지만..)
#전체 데이터셋을 한번 다 실행(학습)시키는 것을 epoch라고 함.
batch_size = 100 # 1 epoch 을 위해 100개씩 나눠 읽음.
#batch size와 관련된 사이트 : https://blog.naver.com/qbxlvnf11/221449595336
num_iterations = int(mnist.train.num_examples / batch_size) #loop는 정수형을 받기 때문.
# num_iterations 는 1 epoch에 필요한 데이터 수를 batch_size로 나누어서
# 몇번을 돌면 1 epoch인지 나타내는 변수.
# 주의!! tf.train과 mnist.train은 다른 train임!!

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs): #루프가 두번돌아간다. for문은 늘 range로 받음을 잊지말기!
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # next_batch함수는 mnist데이터에만 있음. 즉 input_data class에만 있는듯.
            # 1부터 100개 주고, 101번째부터 200까지 주고.. 이런행위임.
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations
            #분모는 550으로 고정임.
            # 100개씩 학습을 시키면서 cost 550종류 다더하고 550 나누는것과 같음.
            # 전체를 한번에 학습하는 cost에서 시행하는 55000으로 나눈것과 다르다.
            # 즉, batch_size를 어떻게 정하느냐에 따라 cost는 달라질 수 있다.

        # 1 epoch이 끝남.

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
        # 4와 9는 자리수를 의미. 이떄 04는 0000을 초기값으로 설정.

    print("Learning finished") #멋잇당

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            #sess.run으로 돌려도 되고, accuracy 같은 tense에 eval이라는 매서드 호출.
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
            #우리의 성능 평가.
        ),
    )
    #간단한 모델임에도 89%로 맞춘다.

    # Get one and predict
    #기본적으로 랜덤하게 하나 읽어옴.
    r = random.randint(0, mnist.test.num_examples - 1)
    #num_examples는 1만개이다.
    #random.randint(최소,최대) = 최소부터 최대까지 중 임의의 정수를 반환함.
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    #array를 행렬인 상태로 유지하면서 r번째를 불러오려면 r:r+1 이렇게 적어줘야 한다.
    #mnist.test.labels[r]도 되지만, 이는 r번째 행을 벡터로 반환해서 불러옴!!

    #테스트 할 label에 1개 읽어옴.
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow( #image show
        mnist.test.images[r : r + 1].reshape(28, 28), #배열 재정의.
        cmap="Greys",
        interpolation="nearest",
        #cmap은 칼라맵, interpolation은 색의 보간처리를 어떻게 할것인지에 대한 내용
        #다음의 사이트를 참고하면 이해할 수 있다.
        #cmap : https://pythonkim.tistory.com/82 cmap의 옵션선택을 할 수 있음.
        #interpolation: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/interpolation_methods.html
    )
    plt.show() #print문과 같음. 습관화하기.


