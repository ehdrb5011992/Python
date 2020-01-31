#Tensor Manipulation
# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

# pprint 과 print의 비교에 대한 설명.
#https://pythonkim.tistory.com/91
# numbers = [[1,2,3],[4,5],[6,7,8,9]]
# print(numbers)
# print(*numbers) #와우!
# print(*numbers,sep='\n')

pp = pprint.PrettyPrinter(indent=4)
#indent는 들여쓰기. 그 칸수를 4칸만큼.
#pp라는 변수에 늘 pprint문법을 적용하고 시행시킨다는 뜻.

sess = tf.InteractiveSession()
#sess의 전달 없이도 tf문의 eval을 실행시킬수 있게 됨.
#참고 : https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/client.html

#비교.
# a = tf.constant(5.0)
# b = tf.constant(6.0)
# c = a * b
# sess = tf.Session()
# sess.run(c) #c.eval() 은 실행불가.
##############
# sess = tf.InteractiveSession()
# a = tf.constant(5.0)
# b = tf.constant(6.0)
# c = a * b
# 'sess'의 전달없이도 'c.eval()'를 실행할 수 있다.
# print(c.eval()) #run을 안쓰고도 쉽게 실행가능.
# sess.close()  #닫을땐 close를 사용해서 끝.

# Simple Array

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
#print(t) 차이점을 봐라. pprint가 더 이쁨.
print(t.ndim) # rank #1차원
print(t.shape) # shape #7개
print(t[0], t[1], t[-1]) #index를 뽑음.
print(t[2:5], t[4:-1]) #slice하는 방법.
print(t[:2], t[3:])


# 2D Array

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]) #1차원 array 섞어놓은거.
pp.pprint(t)
print(t.ndim) # rank 쉽게 계산가능.
print(t.shape) # shape


# Shape, Rank, Axis
t = tf.constant([1,2,3,4])
tf.shape(t).eval() #eval을 쓰는것만으로 값을 출력할 수 있음.

t = tf.constant([[1,2],
                 [3,4]])
#rank는 괄호의 개수, shape 은 ,개수 +1 임
#shape의 원소의 개수가 바로 rank 값임.
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

[ #Axis=0 (0부터 시작해서 안쪽으로 들어간다.)
    [ #Axis=1
        [ #Axis = 2
            [1,2,3,4], #Axis =3 이자 Axis = -1 (제일 안쪽의 있는 값)
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20],
            [21,22,23,24]
        ]
    ]
]
#이쁘게 표현하는 방법임.
#Axis의 개수는 rank임.
#Axis는 0부터 시작하며, 가장 안쪽에 있는 값이 가장 큰 값.



# Matmul VS multiply

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

tf.matmul(matrix1, matrix2).eval()
(matrix1*matrix2).eval() #요소별 곱.


#Watch out broadcasting
#broadcasting이란 , shape이 달라도 연산이 가능하게끔 해주는 기능. (유연하게 계산함)
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval()

matrix1 = tf.constant([[1., 2.]]) #이렇게 행벡터랑 열벡터도 계산이 됨.
matrix2 = tf.constant([[3.], [4.]])
(matrix1+matrix2).eval()
(matrix2+matrix1).eval() #되도록 정확하게 사용하기.

#Random values for variable initializations
tf.random_normal([3]).eval()
tf.random_uniform([2]).eval()
tf.random_uniform([2, 3]).eval()


#Redcue Mean/Sum

tf.reduce_mean([1, 2], axis=0).eval()
#주의!!. int인지 float인지에 따라서 결과값이 다르게나옴. 1.5가 아니다!

x = [[1., 2.],
     [3., 4.]]


tf.reduce_mean(x).eval()
tf.reduce_mean(x, axis=0).eval()
tf.reduce_mean(x, axis=1).eval()
tf.reduce_mean(x, axis=-1).eval() #가장 안쪽에 있는것을 평균내어라.(자주쓴다)

tf.reduce_sum(x).eval()
tf.reduce_sum(x, axis=0).eval()
tf.reduce_sum(x, axis=-1).eval()
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval() #많이 쓰는 코드.


#Argmax with axis

x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()  #위치를 구한다.
tf.argmax(x, axis=1).eval()
tf.argmax(x, axis=-1).eval()

#Reshape, squeeze, expand_dims

t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
t.shape


tf.reshape(t, shape=[-1, 3]).eval() #보통 맨 마지막 값은 그대로 가져가고, 나머지 차원을 조절하기에
                                    #데이터의 의미가 크게 변질되진 않는다.
tf.reshape(t, shape=[-1, 1, 3]).eval()

tf.squeeze([[0], [1], [2]]).eval() #값을 펴줌. 와우! numpy의 flatten()과 같음.
tf.expand_dims([0, 1, 2],1).eval() #역시나 차원을 바꿈.

#One hot

tf.one_hot([[0], [1], [2], [0]], depth=3).eval() #주의! rank가 2에서 3으로 자동으로 됨.
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval() #반드시 one-hot은 reshape와 따라다닌다!
#depth는 필수옵션임.

#만약 y는 벡터이고, tf.one_hot(y)하면, 벡터가 행렬로 바뀌면서 one-hot되므로 상관없음.
#예)
# y = [1,2,3,4]
# tf.one_hot(y,depth=4).eval() 이렇게 design matrix가 만들어진다.

#casting

tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()
#tf.cast([True, False, 1 == 1, 0 == 1]).eval() #얘는 실행이안됨. dtype옵션은 필수기때문.
#True False를 int로 바꾸면 1과 0으로 출력된다!!

#stack

x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
tf.stack([x, y, z]).eval()

tf.stack([x, y, z], axis=1).eval() #축을 바꿈으로써 쌓는 방법을 다르게 할 수 있다.

#Ones like and Zeros like

x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x).eval() #1로 채워진 tense를 x와 같은 shape으로 만들어줌.
tf.zeros_like(x).eval() #얘는 0으로 채워짐.

#Zip
#매우유용하다. 꼭기억
for x, y in zip([1, 2, 3], [4, 5, 6]): #묶어서 한방에 처리 자주쓴다.
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)


#Transpose
# 1. shape 결정 : https://superelement.tistory.com/18 이글을 참고하면 더 도움이 된다.
# 2. 값 결정 : i,j,k 각 원소를 바뀌는 형태(5가지중 하나) 로 1-1 mapping시키면 됨.
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)

t1 = tf.transpose(t, perm= [1, 0, 2]) #변경할 축을 지정함. 옵션은 순열일뿐.
#참고) 기존의 형태 perm은 0,1,2였고 1,0,2로 바꾸라는 의미는
# 0 -> 두번째로, 1 -> 첫번째로, 2 -> 두번째로 라는 의미이다.
# 3차원이면, 3! - 1 =5 개의 perm 가짓수가 존재.
#여기까지가 perm (즉, shape)을 결정짓는 설명.
#[0,1,2]면 , 0에 해당하는 부분이 제일 겉 list임.
#그렇기에, [0,2,1]을 perm 옵션으로 주면 큰 list 형태는 유지하고, 각 행렬이 T됨.

pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))

t = tf.transpose(t1, [1, 0, 2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t)) #다시 원래대로 돌아옴.

t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))

t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))


