#1은 True, 0은 False 나머지는 True취급이나, 정확히 True는 아님.
#이는 R, Python 모두 동일.
0 == True
1 == True
-1 == True
0 == False
1 == False
-1 == False

bool(0)
bool(1)
bool(-1)
bool(-123124124)

bool([])
bool([3]) #값이 하나라도 있으면 true
bool(None)
bool('')
bool("hi")
bool({})

if 'hi' :
    print("hello")

if "" :
    print("hello")

#응용.
value = input('입력해 주세요>')  or '아무것도 못받았어'
print('입력받은값>',value)
#input이 빈 경우, False이 때문에 뒤에가 실행됨.

a = 1 or 10    # 1의 bool 값은 True입니다.
b = 0 or 10    # 0의 bool 값은 False입니다.

print("a:{}, b:{}".format(a, b))