a,b = 1,2
c=(3,4)
c
d,e = c
d
e #c의 값을 unpacking해서 d,e에 넣었다
f=d,e #변수 d와 e를 f에 packing했다.

x=5
y=10
#변수바꾸는 방법(일반적인 경우)
temp=x
x=y
y=temp
#튜프를 이용함.
x,y = y,x #매우 쉽다.

def tuple_func():
    return 1,2
q,w = tuple_func()
q
w

