a=10
if a<0 and 2**a > 1000 and a%5 ==2 and round(a) == a :
    print("복잡한 식")
#처음부터 문제가 발생해서 뒤에 2**a이후는  의미가 없다
#and는 모두 true여야 모두 true
#or는 하나만 true여도 모두 true

def return_false() :
    print("함수 return_false")
    return False

def return_true() :
    print("함수 return_true")
    return True

a=return_false()
b=return_true()

print("테스트1")
if a and b:
    print(True)
else:
    print(False)


print("테스트2")
if return_false() and return_true(): #위와 다른 결과 뒤에 return_ture는
                                     #쳐다보지도 않는다.
    print("True")
else:
    print(False)

print("테스트3")
if return_true() and return_true(): #위와 다른 결과 뒤에 return_ture는
                                     #쳐다보지도 않는다.
    print("True")
else:
    print(False)

#another example
#case1
dic = {"key2" : 'Value1' }
if 'key1' in dic and dic["key1"]== 'value1' :
    print("key1도 있고, 그 값은 value1이네")
else:
    print("아니네") #단락평가후 바로 else로 넘어오게됨.

dic = {"key2" : 'Value1' }
if 'key1' in dic :
    if dic["key1"]== 'value1' :
        print("key1도 있고, 그 값은 value1이네")
else:
    print("아니네") #위는 아래와 같은코드.

#단락평가는 이중,삼중 등의 if문의 효과를 지니고 있음.
#이는 파이썬에서 유용한기능. 돌아가는게 중요하기에, False가 먼저 나오는
#위가 좋은 코드.

#case2
dic = {"key2" : 'value1' }
if dic["key1"] == 'value1' and 'key1' in dic :
    print("key1도 있고, 그 값은 value1이네")
else:
    print("아니네")
#error

