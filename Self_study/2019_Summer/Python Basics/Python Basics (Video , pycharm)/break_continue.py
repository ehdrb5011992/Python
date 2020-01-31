list=[1,2,3,5,6,2,5,237,55]
for val in list:
    if val % 3 == 0 :
        print(val)
        break

# for i in range(10) : #ctrl + / : 전체주석 및 해제
#     if i %2 != 0 :
#         print(i)
#         print(i)
#         print(i)
#         print(i)

for i in range(10) :
    if i%2 == 0 :
        continue #깊은 블록을 들어가면 복잡해보임.
                 #그렇기에, 흥미로운 조건을 만족하면 반복문 처음조건으로
                 #가라라는 뜻.
    print(i)
    print(i)
    print(i)
    print(i)

#continue와 break는 조건문과 함께 씀. if안에서 이루어짐.
#쉽게 break는 반복문을 끝내라, continue는 끝내지는 말고 그냥 위로가라
#라는 뜻. 이는 for문 외에 while문에서도 동작함.
for i in range(10) :
    if i%2 == 0 :
        continue
    print(i)

##########################예제#################
numbers = [ (1,2),(10,0),(20,0),(30,2) ]

for a,b in numbers:
    if b == 0:
        print("0으로 나눌 수는 없습니다.")
        continue
    print("{}를 {}로 나누면 {}".format(a,b,a/b))