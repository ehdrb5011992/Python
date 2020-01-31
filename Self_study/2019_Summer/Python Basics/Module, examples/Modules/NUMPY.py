#numpy
#### 시작하기전에 //
# 클래스 (matrix와 ndarray)의 차이. https://studymake.tistory.com/408
import numpy as np
np.random.seed(0)
a = np.array([2,3,4]) # Not np.array(2,3,4)
print(a)
type(a)
print(a.dtype)
b = np.array([1.2, 3.5, 5.1])
print(b.dtype)

b = np.array([(1.5,2,3), (4,5,6)])
# 위와 같음: b = np.array([[1.5,2,3], [4,5,6]])
print(b)

np.arange(6)

np.arange(1, 10)

np.arange( 0, 2, 0.3) #처음 , 끝, 증가치

np.linspace( 0, 2, 9 )  # 9-1=8 구간임. (9개의숫자) / 결과는 벡터
                        # 0부터 2까지
np.arange(12).reshape(4,3) #벡터를 행렬로 바꿔버림.


A = np.array([[1,1],
              [0,1]])
B = np.array([[2,0],
              [3,4]])

print(A * B) # elementwise product
print(A @ B) # matrix product (in python >=3.5)
print(A.dot(B)) # another matrix product
print(A/B) #요소별 나누기
print(A+B) #요소별 덧셈.
print(np.linalg.inv(A)) #역행렬
print(np.linalg.det(A)) #행렬식

a = np.arange(12).reshape(3,4)
print(a)
print(a.sum()) #전체합
print(a.min()) #전체 최소
print(a.max()) #전체 최대

print(a.sum(axis=0)) # sum of each column -> 열기준
print(a.min(axis=1)) # min of each row -> 행기준
print(a.cumsum(axis=1)) # cumulative sum along each row -> 열기준누적
print(a.cumsum().reshape(3,4)) #전체 누적합구하고, reshape

a = np.arange(10)**2 #요소별 제곱
print(a)
print("***")
print(a[2])
print("***")
print(a[2:5]) # upto position index 4
print("***")
a[:7:2] = -1000 # not equal to a[0:6:2] = -1000
print(a)
print("***")
print(a[ : :-1]) # reversed a
print("***")
for i in a:
    print(i)

b = np.arange(12).reshape(4,3)
print(b)
print("***")
print(b[2,2]) #<- 행렬의 3,3 임
print("***")
print(b[0:4, 1]) # each row in the second column of b
print("***")
print(b[ : , 1]) # equivalent to the previous example
print("***")
print(b[1:3, : ]) # each column in the second and third row of b
print("***")
print(b[-1]) # the last row. Equivalent to b[-1,:]

time = np.linspace(20, 145, 5)                 # time scale
data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series
print(time)
print("***")
print(data)
print("***")
ind = data.argmax(axis=0)                  # index of the maxima for each column
print(ind)
print("***")
time_max = time[ind]                       # times corresponding to the maxima
print(time_max)
print("***")
data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...
                                           # => data.shape : 데이터 차원크기
# =>  data_max = list(data.max(axis = 0)) 와 동치.
print(data_max)
print("***")

a = np.arange(12).reshape(3,4)
b = a > 4
print(a)
print("***")
print(b)                                          # b is a boolean with a's shape
print("***")
a[b] = 0                                   # All elements of 'a' higher than 4 become 0
print(a)
### 즉,
a = np.arange(12).reshape(3,4)
a[a>4]=0
print(a)
#####

a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])             # first dim selection
b2 = np.array([True,False,True,False])       # second dim selection
print(a)
print("***")
print(a[b1,:])                                   # selecting rows
print("***")
print(a[:,b2])                                   # selecting columns

a = np.array(10*np.random.random((3,4))) #(3,4) 매트릭스의 랜덤난수 출력
print(a)
a = np.floor(10*np.random.random((3,4))) #내림함수. np.floor
print(a)
print("***")
print(a.shape)
print("***")
print(a.ravel()) # returns the array, flattened #벡터임 / 1,12는 행렬.
print("***")
print(a.reshape(6,2)) # returns the array with a modified shape
print("***")
print(a.T) # returns the array, transposed
print("***")
print(a.T.shape)
print("***")
print(a.shape)
print("***")
a.resize((2,6))
print(a) #a를 출력함에 주의! a.resize를 출력하지를 않는다!! 이게 reshape와 다른점
print("***")
print(a.reshape(4,-1)) # with -1, the other dimensions are automatically calculated
np.random.seed(0)
a = np.floor(10*np.random.random((2,2))) # 4개의 난수생성 후 행렬로만들어버림. floor은 버림함수
print(a)
b = np.floor(10*np.random.random((2,2)))
print(b)

print(np.vstack((a,b))) #튜플로 받음에 주목.
print("***")
print(np.hstack((a,b)))

a = np.array([4.,2.]) #벡터
b = np.array([3.,8.])
print(np.column_stack([a,b]))     # returns a 2D array 벡터를 붙일때(변수추가) -> column으로 추가
print(np.row_stack((a,b)))     # returns a 2D array 벡터를 붙일때(데어터추가) -> row로 추가
#print(np.hstack((a,b))) #행렬을 붙일때
#print(np.vstack((a,b))) # a와 b를 행렬취급해버리고 붙여버림. = print(np.row_stack((a,b)))
print(a.shape)
print("***")
print(np.hstack((a,b)))           # the result is different
print("***")
print(a[:,np.newaxis])               # this allows to have a 2D columns vector 잊지말기!!!!!
print(a[:,np.newaxis].shape)
#행렬인경우 a = np.array([[1,2],[3,4]]) ; print(a[:,np.newaxis, :])  ; print(a[:,np.newaxis].shape) ???
# a = np.array([[1,2],[3,4]]) ; print(a[:,:,np.newaxis])  ; print(a[:,np.newaxis].shape) ok
print("***")
print(np.column_stack((a[:,np.newaxis],b[:,np.newaxis])))
print("***")
print(np.hstack((a[:,np.newaxis],b[:,np.newaxis])))   # the result is the same

a = np.arange(12)
b = a            # no new object is created
print(a.shape)
print("***")
print(b is a)           # a and b are two names for the same ndarray object
print("***")
b.shape = 3,4    # changes the shape of a -> reshape랑 다름.
                 # 이는 resize와 같음. b.resize(3,4)
print(b)
print("***")
print(a.shape) #즉, b를 resize시키면 a도 resize 된다.


d = a.copy()  # a new array object with new data is created
print(d is a) # a와 d는 다르기에, copy함수를 쓰는것이 바람직. 이는 numpy의 내장함수는 아님.
print("***")
d[0,0] = 9999
print(d)
print("***")
print(a) #a와 d는 다르다.

a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
print("***")
print(a.transpose()) # print(a.T)
print("***")
print(np.linalg.inv(a)) #linear algebra inverse matrix

u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
print(u)
print("***")
print(np.trace(u))  # trace

j = np.array([[0.0, -1.0], [1.0, 0.0]])
print(j)
print("***")
print(j @ j)        # matrix product

y = np.array([[5.], [7.]])
print(np.linalg.solve(a, y))
# solve(a, b)
# 행렬 방정식을 해결한다.
# a * x = b 일 경우 x의 값을 구한다.