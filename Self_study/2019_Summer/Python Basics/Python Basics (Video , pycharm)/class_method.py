class Human():
    '''인간'''
def create_human(name,weight) :
    person =Human()
    person.name = name
    person.weight = weight
    return person
Human.create = create_human #휴먼이라는 클레스 안에 create라는 매소드로
                            #create_human함수를 저장.

def eat(person) :
    person.weight +=0.1
    print('{}가 먹어서 {}kg이 되었습니다.'.format(person.name,person.weight))

def walk(person) :
    person.weight -= 0.1
    print('{}가 걸어서 {}kg이 되었습니다.'.format(person.name, person.weight))

Human.eat = eat #휴먼에 관련된 함수이기에 넣어서 쓰면 편하다
Human.walk = walk
### 이 작업이 번거롭다.
### 클레스 안에 함수를 바로 적용하는 기능을 제공함.
### 아래와 같이 합칠 수 있음.
################################################################

class Human():
    '''인간'''
    def create(name, weight): #self가 아니다. name이라는 변수로 받는것. 이런경우는 아래와 같이 '철수'를 그대로 받음.
        person = Human()
        person.name = name  #person이라는 class의 name이라는 변수에 def에서 사용한 name변수를 저장.
        person.weight = weight
        return person #라고 정의하는 사람을 돌려줌. 이게 __init__의 역할을 하는것.

    def eat(hi): #person대신 hi로씀. 이렇게 각 함수 내에서는 그냥 변수를 따로따로 저장해도 상관없음.
        hi.weight += 0.1
        print('{}가 먹어서 {}kg이 되었습니다.'.format(hi.name, hi.weight))

    def walk(person): #위랑 같은거임.
        person.weight -= 0.1
        print('{}가 걸어서 {}kg이 되었습니다.'.format(person.name, person.weight))

donggyu = Human.create('동규',60.5) #바로 매서드로서 사용. 이때 create는 return을 person이라는 Human class로 돌려주기 때문에 donggyu라는 변수는 Human class이다.
#즉, donggyu.weight = 60.5 , donggyu.name = '동규' 로 저장이 된 상황임!!

donggyu.walk() #method를 호출한다. #person.walk로 쓰면 이름이나옴. 함수이기 때문에 ()로 정해줌.
#walk(person) 는 정의안됨.
donggyu.eat()
donggyu.walk() #클래스로 현실의 개념을 표현하는 것을 모델링이라고 한다.


#self변수에 대해 알아보자.

class Human():
    '''인간'''
    def create(name, weight):
        person = Human()
        person.name = name
        person.weight = weight
        return person  #얘는 예외로 지금은 다루지 않음.

    def eat(self): #이렇게 써도 상관없음. self대신에 아무거나 와도 상관없음.
        self.weight += 0.1
        print('{}가 먹어서 {}kg이 되었습니다.'.format(self.name, self.weight))

    def walk(self): #이렇게 써도 상관없다.
        self.weight -= 0.1
        print('{}가 걸어서 {}kg이 되었습니다.'.format(self.name, self.weight))

    def speak(self,message): #이렇게 써줘야함.
        print(message)


 # def eat(self):
 #        self.weight += 0.1
 #        print('{}가 먹어서 {}kg이 되었습니다.'.format(self.name, self.weight))

person = Human.create('철수',60.5) #바로 매서드로서 사용.
#eat() #얘는 안됨.
person.eat() #애는 class안에 있는 것으로 호출됨.
#person이라는 변수가 eat의 첫번째 매개변수인 self로 자동으로 전달됨.
#즉, person.eat(a1,a2) 는 3개의 매개변수가 있었던거임.

person.speak("안녕하세요.")


