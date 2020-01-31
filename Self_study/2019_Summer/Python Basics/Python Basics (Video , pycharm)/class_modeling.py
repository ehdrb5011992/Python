class Human():
    '''인간'''
# person = Human()
# person.name = '철수'
# person.weight = 60.5

def create_human(name,weight) :
    person =Human()
    person.name = name
    person.weight = weight
    return person

Human.create = create_human #휴먼이라는 클레스 안에 create라는 매소드로
                            #create_human함수를 저장.

person = Human.create('철수',60.5) #이렇게도 쓸 수 있음.

def eat(person) :
    person.weight +=0.1
    print('{}가 먹어서 {}kg이 되었습니다.'.format(person.name,person.weight))

def walk(person) :
    person.weight -= 0.1
    print('{}가 걸어서 {}kg이 되었습니다.'.format(person.name, person.weight))

Human.eat = eat #휴먼에 관련된 함수이기에 넣어서 쓰면 편하다
Human.walk = walk

person.walk()
person.eat()
person.walk() #클래스로 현실의 개념을 표현하는 것을 모델링이라고 한다.

