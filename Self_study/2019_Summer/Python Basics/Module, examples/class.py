
import numpy as np
x = np.array([[1,2],[10,20]])
y = np.array([[0.01,0.02],[0.1,0.2]])
z = [[sum([x[i][k]*y[k][j] for k in range(2)]) for i in range(2)] for j in range(2)]

for i in range(5):
    x = 1
    y = 3/x
x

for i in range(5):
    try:
        x = 0
        y = 3/x
    except Exception:
        print("Something wrong")


import math
class Vector2:
    def __init__(self,x,y):
        self._x = x
        self._y = y
    def x(self): #x를 class로 쓰기 위해서 정의
        return self._x
    def y(self): #y를 class로 쓰기 위해서 정의
        return self._y
    def size(self):
        return math.sqrt(self.x()**2 + self.y()**2)
v1 = Vector2(1,1)
print(v1.size())
####################expansion############
import math
class Vector2:
    def __init__(self,x,y):
        self._x = x
        self._y = y
    def x(self):
        return self._x
    def y(self):
        return self._y
    def __sub__(self, other): #뺄셈연산.
        return Vector2(self.x() - other.x(),
                       self.y() - other.y())
    def __repr__(self): #표현연산.
        return "Vector2({},{})".format(self.x(), self.y())
    def size(self):
        return math.sqrt(self.x() ** 2 + self.y() ** 2)

v1 = Vector2(1, 0)
v2 = Vector2(0, 1)
v = v1 - v2
print(v)
print(v.size())

####################expansion2############
import math
class Vector2:
    def __init__(self,x,y):
        self._x = x
        self._y = y
    def x(self):
        return self._x
    def y(self):
     return self._y
    def __sub__(self,other):
        return Vector2(self.x() - other.x(),
                       self.y() - other.y())
    def __repr__(self):
        return "Vector2({},{})".format(self.x(),self.y())
    def size(self):
        return math.sqrt(self.x()**2 + self.y()**2)

class Point2(Vector2): #vector2 의 성질들은 다 가지고 있음. - 계승이기에
    def __init__(self,x,y):
        super().__init__(x,y)
    def distance(self,other):
        v = self - other
        return v.size()
p1 = Point2(1,0)
p2 = Point2(0,1)
print(p1.distance(p2))



class Vector:
    def __init__(self,lst,dim=None):
        self._vec = list(lst)
        if dim is None:
            self._dim = len(self._vec)
        else:
            self._dim = dim
    def dim(self): #vector.dim()을 보기위한 함수.
        return self._dim
    def __repr__(self):
        return "Vector({0})".format(self._vec)
    def __str__(self): #str, repr 둘다 문자열을 반환함.
                       #str은 추가적가공,다른데이터와 호환,
                       #repr은 단순히 표현하는데 목적이있음.
                       #결론 : str이 훨씬 유용함.
        if (self.dim() > 3):
            return "Vector({},{},...)".format(self[0],self[1]);
        else:
            return self.__repr__()
    def __getitem__(self,key): #[]같은 명령어로 값을 얻고싶을때
        return self._vec[key]
    def __setitem__(self,key,val): #튜플이아닌 list처럼 값을 수정하고 싶을때
        self._vec[key] = val
    def __add__(self,other): #더하는 연산
        return Vector([self[i] + other[i] for i in range(self.dim())])
    def __sub__(self,other): #빼는 연산
        return Vector([self[i] - other[i] for i in range(self.dim())])
    def size(self):
        return math.sqrt(sum([x*x for x in self._vec]))

x = Vector(range(100))
y = Vector(range(100))
x[0] = 10
print('x-y=', x - y)
print('x+y=', x + y)
print('x[10]=',x[10])
print('(x-y).size()=',(x-y).size())


############ final ##########

class Point(Vector) :
    def __init__(self,lst):
        super().__init__(lst)
    def distance(self,other):
        elements = self - other
        return elements.size()

p1 = Point(range(100))
p2 = Point(range(100))
p1[0] = 10
print("p1.distance(p2)=",p1.distance(p2))


##############str과 repr의 차이##############
class A:
    def __str__(self):
        return 'str method is called'
    def __repr__(self):
        return 'repr method is called'
a=A()
repr(a)
str(a)
a
print(a)
list(a) # 불가
list(str(a)) #가능
##############str과 repr의 차이##############
