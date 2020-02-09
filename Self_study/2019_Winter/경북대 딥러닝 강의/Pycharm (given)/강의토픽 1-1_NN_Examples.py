"""
참고교재: TensorFlow 2.0 Quick Start Guide (Holdroyd 저)

예제 1-1. 신경망 관련 4개의 Keras 모형 
"""


#1. pip install --user tensorflow-gpu
#2. pip3 install --user --upgrade tensorflow--gpu
#3. pip install --user --ignore-installed --upgrade tensorflow-gpu

# print(tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None))
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

#--------------------------------------------------------------------
# (1) Keras Sequential Model (첫번째)
#--------------------------------------------------------------------
import tensorflow as tf
#from tensorflow import keras
#import numpy as np

(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

epochs=10
batch_size = 32 # 32 is default in fit method but specify anyway (미니배치 사이즈)

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
#tf.cast는 텐서플로우에서 강제로 타입을 바꾸는것임. 이렇게 바꿔야 돌아간다.
#만약 범주를 나타내는거면 아래처럼 정수를 그대로 쓰면 된다.
train_y, test_y = tf.cast(train_y,tf.int64),tf.cast(test_y,tf.int64)

#sequential api는 내가 구성하고싶은 모형대로 차곡차곡 쌓으면 됨.
model1 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(), #flatten이라는 함수를 통해 평평하게 만들어놓음. (MLP의 입력층)
  tf.keras.layers.Dense(512,activation=tf.nn.relu), #노드갯수 512개, act.ftn은 ReLU
  tf.keras.layers.Dropout(0.2), #20%의 input을 0으로 둔다. (80%만 사용)
  tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

optimiser = tf.keras.optimizers.Adam() #Adam 최적화. 디폴트값을 주고 사용. Adam클래스에서 초기값을 줘서 따로 정의 가능.
model1.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
# sparse_categorical_crossentropy 는 one-hot encoding을 안하고 바로 접근하는 방법.
# categorical_crossentropy를 쓰려면 one-hot encoding을 해야함.
# 정확도를 보고자 할 때 측도를 'accuracy로 줌.

model1.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

model1.evaluate(test_x, test_y)
#만약 test 의 x값만 있고, y값이 없다면 predict라는 함수를 써줘야함.

#--------------------------------------------------------------------
# (2) Keras Sequential Model (두번째)
#--------------------------------------------------------------------
import tensorflow as tf
#from tensorflow import keras
#import numpy as np

(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

epochs=10
batch_size = 32 # 32 is default in fit method but specify anyway

train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
train_y, test_y = tf.cast(train_y,tf.int64),tf.cast(test_y,tf.int64)

#여기부분만 달라진다.
model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(512, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.2))
model2.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
#add를 통해서 층을 게속 만들어나감.
#간단한 문제를 풀고자 할 때 위처럼 구현하면 편하다.
#참고로, classification을 진행할 때는 데이터가 balance data이어야 한다.
#이에 대해서 이미지의 경우에는 GAN을 사용하는 방법도 좋다.

model2.compile (optimizer= tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',
  metrics = ['accuracy'])

model2.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
# 위의 참고와 관련되는 내용으로, 위의 'fit' method에서 옵션을 주어 imbalanced data를 balanced data로
# 바꿀 수 있다. (unbalanced를 고려한 가중치를 주게 됨.)

model2.evaluate(test_x, test_y)

###### keras를 사용할 것이면, 얘를 눈여겨 보자!! ########
# api는 sequential버전이 있고, functional 버전이 있음.
#--------------------------------------------------------------------
# (3) Keras functional API (세번째)
#--------------------------------------------------------------------
import tensorflow as tf
#from tensorflow import keras
#import numpy as np

(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x/255.0, test_x/255.0
epochs=10

inputs = tf.keras.Input(shape=(28,28)) # Returns a 'placeholder' tensor
x = tf.keras.layers.Flatten()(inputs) #마치 함수구조로 쌓아간다. input을 x로 하자.
x = tf.keras.layers.Dense(512, activation='relu',name='d1')(x) #앞에서 차근 차근 쌓아감.
x = tf.keras.layers.Dropout(0.2)(x) #앞에서 차근 차근 쌓아감.
predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')(x) #앞에서 차근 차근 쌓아감.
# 좀 더 복잡한 모형을 쌓으려고 한다면, 이렇게 하는것을 추천한다.

model3 = tf.keras.Model(inputs=inputs, outputs=predictions)

model3.summary()

optimiser = tf.keras.optimizers.Adam()
model3.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model3.fit(train_x, train_y, batch_size=32, epochs=epochs)

model3.evaluate(test_x, test_y)


# 얘는 교수님 개인적으로는 별로 추천하지 않는데 이런것도 있다.
# Subclassing model
#--------------------------------------------------------------------
# (4) Subclassing Keras Model Class (네번째)
#--------------------------------------------------------------------
import tensorflow as tf
#from tensorflow import keras
import numpy as np

(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x/255.0, test_x/255.0
epochs=10

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
    # Define your layers here.
        inputs = tf.keras.Input(shape=(28,28)) # Returns a placeholder tensor
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(512, activation='relu',name='d1')
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')

    def call(self, inputs):
    # This is where to define your forward pass
    # using the layers previously defined in `__init__`
        x = self.x0(inputs) #여기 구조는 functional api처럼 생겼다.
        x = self.x1(x)
        x = self.x2(x)
        return self.predictions(x)

model4 = MyModel()

batch_size = 32
steps_per_epoch = int(len(train_x)/batch_size)
print(steps_per_epoch)

model4.compile (optimizer= tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',
  metrics = ['accuracy'])

model4.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

model4.evaluate(test_x, test_y)

model4.summary()


# 교수님이 개인적으로 선호하는 방식.
#--------------------------------------------------------------------
# (5) 일반적 Tensoeflow 2.0을 이용한 다층 신경망 (다섯번째)
#--------------------------------------------------------------------
# 필요한 라이브러리를 불러들임
import tensorflow as tf

# MNIST 파일 읽어들임
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.cast(x_train.reshape(60000,784), tf.float32)/255.0   # rshaping 및 rescaling
x_test = tf.cast(x_test.reshape(10000,784), tf.float32)/255.0     # rshaping 및 rescaling

# 클래스 변수 y를 one-hot 벡터로 만듦.
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,10)   #dtype=float32
y_test = to_categorical(y_test,10)     #dtype=float32

# 학습 관련 매개변수 설정 
n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_output = 10

lr_rate = tf.constant(0.001,tf.float32)

## 가중치 및 편향 초기화: Glorot 방법 사용
glorot_init = tf.initializers.glorot_uniform(seed=42)

# 은닉층 1
w_h1 = tf.Variable(glorot_init((n_input, n_hidden1)))
# 은닉층 2
w_h2 = tf.Variable(glorot_init((n_hidden1, n_hidden2)))
# 은닉층 3
w_out = tf.Variable(glorot_init((n_hidden2, n_output)))

# 편향 1, 2 및 3 초기화
b1 = tf.Variable(glorot_init((n_hidden1,)))
b2 = tf.Variable(glorot_init((n_hidden2,)))
b_out = tf.Variable(glorot_init((n_output,)))

## 가중치 및 편향을 하나로 묵음 
variables = [w_h1, b1, w_h2, b2, w_out, b_out]  

## 학습 관련 함수 정의하기  
def feed_forward(x):
    # layer1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w_h1), b1))
    # layer2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, w_h2), b2))
    # output layer
    output = tf.nn.softmax(tf.add(tf.matmul(layer2, w_out), b_out))
    return output

def loss_fn(y_pred, y_true):
#    loss = tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[1]))
    return loss

def acc_fn(y_pred, y_true):
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    y_true = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
    predictions = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    return tf.reduce_mean(predictions)

def backward_prop(batch_xs, batch_ys):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    with tf.GradientTape() as tape: #미분을 해야하는걸 tf.GradientTape() 로 저장해나감. with함수로 받음.
        predicted = feed_forward(batch_xs)
        step_loss = loss_fn(predicted, batch_ys)
    grads = tape.gradient(step_loss, variables)
    optimizer.apply_gradients(zip(grads, variables)) #미분을 할 때, zip 함수를 같이 사용해야함.
#여기 까지 해야지 미분이 일어남. 

#----------------------------------------------------------------------
# 신경망을 학습하는 과정
#----------------------------------------------------------------------
n_epochs = 20
batch_size = 128
total_batch = int(len(x_train)/batch_size)
n_shape = x_train.shape[0]   #또는 len(x_train) 사용
no_steps = n_shape//batch_size
display_step = 1

for epoch in range(n_epochs):
    avg_loss = 0.
    avg_acc = 0.
    for i in range(total_batch):
        batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
        
        pred_ys = feed_forward(batch_xs)
        avg_loss += float(loss_fn(pred_ys, batch_ys)/no_steps) 
        avg_acc += float(acc_fn(pred_ys, batch_ys) /no_steps)
        backward_prop(batch_xs, batch_ys)


    if epoch % display_step == 0:        
        #print('Epoch: {epoch}, Training Loss: {avg_loss}, Training ACC: {avg_acc}')
        print("Training loss and Accuracy after epoch {:02d}: {:.4f} and {:.8f}".format(epoch, avg_loss, avg_acc))
        #Training loss and Accuracy after epoch 19: 0.0488 and 0.99151979 

print("Neural Network Training Completed!")        


#----------------------------------------------------------------------
# 검정 데이터에 대해서  싱경망의 Accuray를 구함
#----------------------------------------------------------------------

test_pred_y = feed_forward(x_test) 
test_accuracy = acc_fn(test_pred_y, y_test)

print("Accuracy for test data:  {:.8f}".format(test_accuracy))  #Accuracy for test data:  0.97310001


"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)

예제 1-2. UCI 기계학습 저장소에 있는 Wiine 데이터를 이용하여 은닉층이 1개인 
MLP 분류 알고리즘을 작성하시오. 이때 종속변수는 첫 번째 열에 있는 Wiine의 
품종을 나타내는 class 변수이다
(자료 위치: http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data).
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 데이터 불러오기
col_name = ["Class", "Alcohol", "Malic_acid","Ash", "Alcalinity_of_ash", 
"Magnesium","Total_phenols", "Flavanoids","Nonflavanoid_phenols",
"Proanthocyanins","Color_intensity","Hue","Diluted wines","Proline"]

wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None,
                 names=col_name,na_values=["NA", "null", ""],sep=",")
wine.info()
wine.describe()
pd.isnull(wine).sum() # 결측값 없음 

wine.Class.value_counts() # class 변수 1 2 3 -> 0 1 2
wine.Class # class 1 2 3 
wine.Class = wine.Class-1 # class 0 1 2

# Wine 데이터의 입력 행렬
wineX = wine[col_name[1:]]

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(wineX, wine.Class, test_size=0.3, random_state=20200128)
x_train.shape
y_train.shape

# MinMax 표준화
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) #Compute the minimum and maximum, then transform it.
x_test = scaler.transform(x_test)

#===== 1개 은닉층을 갖는 신경망 모형의 은닉노드 개수 결정 (start)  =====#
# 즉, 모델 (model) selection을 하고 싶을 때, 아래의 코드를 보면됨. 이 작업은 시간이 굉장히 오래걸림.
# 참고: Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
def build_model(n_neurons=30):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_neurons,activation="tanh",input_shape=[13])) # 은닉층은 한개 
    model.add(tf.keras.layers.Dense(3,activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"]) 
    return model

"""
참고: Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition

def build_model2(n_hidden=1,n_neurons=30,learning_rate=3e-3,input_shape=[13]):
    model=tf.keras.models.Sequential()
    options={"input_shape":input_shape}
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons,activation="tanh",**options))
        options={}
    model.add(tf.keras.layers.Dense(3,activation="softmax"))
    optimizer=tf.keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"]) 
    return model
"""

keras_class = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)

from sklearn.model_selection import RandomizedSearchCV

para_distribs={
        "n_neurons":np.arange(1,50) # reciprocal 상호 연속적인 랜덤변수 또는 역수 
}

from sklearn.model_selection import GridSearchCV
search_cv = GridSearchCV(estimator=keras_class, param_grid=para_distribs,
                  scoring='accuracy', cv=3)
search_cv.fit(x_train, y_train, validation_data=(x_test,y_test))



search_cv = RandomizedSearchCV(keras_class,para_distribs,n_iter=10,cv=3)
search_cv.fit(x_train,y_train,epochs=100,
              validation_data=(x_test,y_test),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

search_cv.best_params_ 
search_cv.best_score_ 
model = search_cv.best_estimator_.model # save model

model.fit(x_train, y_train, epochs=100)
result = model.evaluate(x_test,y_test)

print("loss:",result[0]) # 0.1158
print("accuracy:",result[1]) # 0.9815
#===== 1개 은닉층을 갖는 신경망 모형의 은닉노드 개수 결정 (end)  =====#

#===== 위와 동일한 결과인지 확인하기 =====#
model_con = tf.keras.models.Sequential([
        tf.keras.layers.Dense(37,activation="tanh",input_shape=[13]),
        tf.keras.layers.Dense(3,activation="softmax")
        ])
    
model_con.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_con.summary()

model_con.fit(x_train, y_train, epochs=100)

model_con.evaluate(x_test,y_test)


"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하 공저)

예제 1-3. UCI 기계학습 저장소에 있는 Boston 집값 데이터를 활용하여 
집값(MEDV)을 출력변수로 하는 신경망 회귀모형을 생성하여 검증용 자료에 회귀모형의 
성능을 평가하시오. 평가는 housing 데이터 중 훈련자료 70%와 검정자료 30%로 
분할하여 검정하시오(random.state=200으로 고정).
"""

# 변수명 설정
boston_name = ["CRIM","ZN","INDUS",
          "CHAS","NOX","RM","AGE","DIS",
          "RAD","TAX","PTRATIO",    
          "B", "LSTAT", "MEDV"]

# 데이터 불러오기
boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
     header=None, sep="\s+", names=boston_name, na_values=["NA", "null", ""])

# 결측값 제거
pd.isnull(boston).sum()
boston1 = boston.dropna()
boston1.dtypes
# 종속변수 설정
y4 = boston1["MEDV"]

x_train4, x_test4, y_train4, y_test4 = train_test_split(boston1[boston_name[0:-1]], y4, test_size=0.3, random_state=200)

# MinMax 표준화
minmax4 = MinMaxScaler()
x_train4 = minmax4.fit_transform(x_train4)
x_test4 = minmax4.transform(x_test4) 


def build_model4(n_hidden=1,n_neurons=30,input_shape=[x_train4.shape[1]]):
    model = tf.keras.models.Sequential()
    options = {"input_shape":input_shape}
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons,activation="relu",**options))
        options={}
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse",optimizer=tf.keras.optimizers.RMSprop(0.001),metrics=["mae","mse"]) # mae 는 뭐?
    return model

para4={
      "n_hidden":np.arange(1,10),
      "n_neurons":np.arange(1,100)
      }

keras_class4 = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model4)

search_cv4_1 = RandomizedSearchCV(keras_class4,para4,n_iter=10,cv=3)
search_cv4_1.fit(x_train4,y_train4,epochs=100,
              validation_data=(x_test4,y_test4),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

search_cv4_1.best_params_ # 5hidden 70 neuron
search_cv4_1.best_score_
model4_1 = search_cv4_1.best_estimator_.model

# 검정용 데이터 적용
pred4_1 = model4_1.predict(x_test4)
result4_1 = model4_1.evaluate(x_test4, y_test4)
print("MSE:",result4_1[0]) # 18.88
print("MAE:",result4_1[1]) # 3.11

# 한 번 더 
search_cv4_2 = RandomizedSearchCV(keras_class4,para4,n_iter=10,cv=3)
search_cv4_2.fit(x_train4,y_train4,epochs=100,
              validation_data=(x_test4,y_test4),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

search_cv4_2.best_params_ # 6hidden 36 neuron
model4_2 = search_cv4_2.best_estimator_.model

# 검정용 데이터 적용
pred4_2 = model4_2.predict(x_test4)
result4_2 = model4_2.evaluate(x_test4, y_test4)
print("MSE:",result4_2[0])
print("MAE:",result4_2[1])

