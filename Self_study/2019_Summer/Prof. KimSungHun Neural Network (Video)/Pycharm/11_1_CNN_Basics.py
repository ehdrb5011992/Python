# ***** output = floor( (N-F+2P ) / S ) + 1  이다. (짝수, 홀수 관계없이) *****
# ***** padding = ceil( (F-1)/2 ) (짝수, 홀수 관계없이) *****
# 잊지말기.

#%matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession() #sess.run 없이 eval이라는 매서드를 통해 자체실행 가능.
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)
# 1 , 3 , 3 , 1
# 1 <- 데이터의 개수
# 3,3 <- 3*3 짜리
# 1 <- 색깔텀 (depth)
# (n , p , q , depth ) 이런 느낌. n은 batch_size라고 하는것이 더 정확하다.
# 위의 사실은 코딩을 쉽게하기 위해 미리약속한 것이므로, 받아들이기.

print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')


#1 filter (2,2,1,1) with padding: VALID
# print("imag:\n", image)
print("image.shape", image.shape)
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]]) #필터 값. 원소별 곱해서 더하는거임.
# 2 , 2 , 1 , 1
# 2, 2 <- 2*2 짜리
# 1 <- 색깔텀 (depth)
# 1 <- 필터의 수
# ( p , q , depth , filters ) 이런 느낌.
# 위의 사실은 코딩을 쉽게하기 위해 미리약속한 것이므로, 받아들이기.

print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
#이 함수 하나로 모든것이 끝남.

#"VALID" 옵션은 stride만큼 읽으면서 남게되는, 행렬의 맨 오른쪽과 맨 아래를 버리라는 뜻.
# 참고 : https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t

#stride는 1*1 짜리임. 리스트의 0번째와 3번째의 1은 지켜주자. 또한,
#1번째와 2번째는 보통 같은수의 값을 넣는다.
#이건 내 느낌인데, 이렇게 strides를 하는 이유는 계산할 때 차원을 맞춰주기 위함이고,
#맨 앞의 1은 데이터와 연동되는칸, 맨뒤의 1은 depth과 연동되는 칸으로 생각된다....?
#즉, 존재의 의미를 위해 차원을 늘리고, 항등원인 곱셈의 항등원인 1을 넣어주는것 같음.

conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
#shape는 (1,2,2,1)이 나오며, 이는 (n,p,q,col)로 바라볼 수 있다.

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
#Transpose와 같이 축을 바꾸라는 명령어. 이때, 0과 3은 축번호에 해당함.
#가장 바깥의 축이 axis = 0 -> 이후 안쪽으로 들어갈수록 axis는 1씩 증가함.

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
#계산을 볼 수 있다.

#2 filter (2,2,1,1) with padding:SAME
# print("imag:\n", image)
print("image.shape", image.shape)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
#잊지말것. Weight는 필터임.
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
#원래 이미지 size와 같게 해주겠다는 뜻이  Padding = Same 임.
#그러면, 자동으로 맞게 필요한만큼 0으로 채움. 이러면 input과 ouput의 차원이 같아짐.
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
#1,3,3,1 을 확인할 수 있다.
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')



#3 filters (2,2,1,3)
# print("imag:\n", image)
print("image.shape", image.shape)

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]]) #필터 3개
# 2, 2, 1, 3
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
#3장의 이미지가 나온다.
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
#(1,3,3,3) 임. 이때는 (n,p,q,depth) 에서 depth이 기존의 color가 아닌, 필터의 개수로 들어오게 된다.
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
print("conv2d_img.shape", conv2d_img.shape)
#(3,3,3,1) 이 나옴.
#원래 conv2d (1,3,3,3)는 3x3짜리 가 3개의 필터를 가지고 1개짜리로 있었다. 이를 0번째와 3번째로 바꾸는 이유는,
#아래의 사진을 출력할때 인덱스를 필터개수당으로 하기 위해서 그럼. (3,3,3,1)은 3개짜리 필터가 3x3x1 을 지닌채로 들어온다는 뜻이됨.
#그래서 아래의 for문에서는 3개의 결과가 나오게 된다.

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
#루프는 3번돔. (i,one_img)가 튜플로 같이도는거임.


#MAX POOLING
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32) #주어진 이미지.
#(1,2,2,1)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
#max_pool이라는 함수를 사용함. CNN과 연동이 잘됨.
#역시나 pooling에서도 strides와 padding이 필요함.
#ksize는 필터로 쓰게될 차원의 크기이며,
#[1,2,2,1]이란 strides와 똑같이 적용된다. 2*2짜리를 1개의 batch_size에 대해 depth는 1짜리로 적용할거라는 뜻.
#거의 어지간하면 양 끝은 1,1로 주게 된다.


print(pool.shape)
print(pool.eval())

#SAME: Zero paddings
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
#image.shape 은 (1,2,2,1)이다. 위의 shape이 1,2,2,1이라는 말.
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print(pool.shape)
#(1,2,2,1) 이 나온다. 즉, zero padding을 하면 image와 같은 결과임. 당연하다.
print(pool.eval())

###########실전 데이터에 넣어봐서 사용해봄. ###########

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
img = mnist.train.images[0].reshape(28,28) #데이터의 크기는 알려져있다.
#mnist.train.images[0].shape 는 784차원 벡터이다.
plt.imshow(img, cmap='gray')


sess = tf.InteractiveSession()


img = img.reshape(-1,28,28,1) #-1은 니가 알아서 계산해 라는 뜻. 사실 알고있으므로, -1대신 1을 써도 상관없다.
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# 3x3 x 1개짜리 칼라(깊이)/ 짜리 5개의 필터를 사용하겠음.
#위의 명령어는 초기값으로 (3,3,1,5)의 shape을지닌 랜덤난수를 생성하라는 뜻. 이때 표준편차는 0.01

conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
#stride 는 2*2 짜리를 씀.
# 14* 14짜리로 출력됨. stride = 2기 때문. (28-3)/2 +1
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval() #값으로 반환해서 저장. 이게 가능하는 이유는 위에서 tf.InteractiveSession()을 했기 때문.
conv2d_img = np.swapaxes(conv2d_img, 0, 3) #역시나 축을 바꿔주고 봐보자.


for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
#이 과정 출력.
#5개의 조금씩 다른 이미지를 convolution으로 뽑아냄.

#이후 maxpooling을 시행해봄.
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')
# 7* 7짜리로 출력됨. stride = 2 (가로,세로전부), filter = 2 (가로,세로전부) 이기 때문.

print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')

#subsampling 된것을 확인.
#이런 행위들이 정보를 몰아주는 행위인 것이다.