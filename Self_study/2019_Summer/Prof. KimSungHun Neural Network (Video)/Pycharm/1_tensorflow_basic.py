import tensorflow as tf
tf.__version__

#Hello TensorFlow!
# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, TensorFlow!")
#한개의 노드가 Hello, TensorFlow! 이름을 지니고 만든거임.
# start a TF session
sess = tf.Session() #그리고 Session을 통해서 (Session Class로 할당해버림.)
print(sess.run(hello)) #실행까지.
#이때 b는 byte string으로 크게 신경쓸 부분은 아님.


#Computational Graph
node1 = tf.constant(3.0, tf.float32) #3.0이라는 노드를 하나 만듦.
node2 = tf.constant(4.0) # also tf.float32 implicitly #4.0이라는 노드를 하나 만듦.
node3 = tf.add(node1, node2) #더하기 노 드를 만듦. 이때, 계산이 되는게아니라 그 자체의 연산을 정의한것.
#node3 = node1 + node2 #로 해도 무방

print("node1:", node1, "node2:", node2)
#결과값은 나오지 않고, 그냥 형태만 말해줌.
print("node3: ", node3)
sess = tf.Session() #결과를 실행시키려면, Session을 하고,
print("sess.run(node1, node2): ", sess.run([node1, node2])) #run을 통해서 실행시켜야함.
#즉, Session은 그래프를 인자로 받아서 실행해주는 일종의 runner로 생각하면 됨.
print("sess.run(node3): ", sess.run(node3))

#즉 우리는 그래프를 build하고, Session.run을 통해서 그래프를 실행시키고, 그 결과로
#값들을 업데이트 하거나 계산하는 형태를 갖게된다.

#Placeholder
#일종의 함수 형태임. 어느 데이터든 유용하게 받아들이기 위해 쓰는 방법임.
#주로 얘를쓸꺼임.
a = tf.placeholder(tf.float32) #노드를 placeholder라고 만들어 줌.
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b) #그리고 add_node를 만듦.

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5})) #값을 넘겨주면서 그래프를 실행시키면 7.5가 출력
#이때 feed_dict이라는 사전형태를 초기값으로 줌.
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))
#이렇게 복수로 줄 수도 있으며, 각각이 더해지는 형태임.

#즉, placeholder로 만들게 되면 위처럼 사용해서도 구할 수 있음.
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b:4.5}))


#Tensors
#텐서는 데이터를 이야기하고, Rank, Shapes, Types를 가짐.
3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
#[2,3] 에서 2는 전체괄호 내부에서 처음으로 맞이하는 괄호의 갯수, 3은 그 하위 항목의 갯수임.
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
#얘를 보고 rank= 3, shape = [2 1 3] , type은 float32임을 알 수 있어야함.
#각 괄호에서 콤마개수 +1 -> shape , 괄호의 개수 ->  rank , 각 숫자의 type


#대부분의 경우 type으로 tf.float32 / tf.int32를 씀.

x = [4,3,2,1]