class Human():
    ''''사람'''

person1 = Human()
person2 = Human()

a=list()
isinstance(a,list)

isinstance(person1,Human)

list1=[1,2,3]
list1.append(4)

person1.language = '한국어'
person2.language = 'English'

print(person1.language)
print(person2.language)

person1.name = '서울시민'
person2.name = '인도인'

#보기 좋게 포장할수 있기 때문에 클래스를 사용. 특별한 기능이 더있는건아님
#행동을 클레스 안으로 넣어서 판단할 수 있다.

def speak(person) :
    print("{}이 {}로 말을 합니다.".format(person.name,person.language))

speak(person1)
speak(person2)
####### 이게 중요.
Human.speak = speak #휴먼 클래스는 말할 수 있는 능력이 생김.
####### 이게 중요.
person1.speak() # = speak(person1)
person2.speak() # = speak(person2) , method로 바꿔버림.
