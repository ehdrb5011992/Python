class Animal(): #부모클래스
    def walk(self):
        print('걷는다')
    def eat(self):
        print('먹는다')
    def greet(self):
        print("인사한다") #똑같은 매서드를 각각에 추가하므로써
                         # 각각의 greet함수를 적용시킴 이를 override개념.
class Cow(Animal) :
    '''소'''

class Human(Animal): #상속의 개념. Human클래스는 자식클래스
    def wave(self):
        print('손을 흔든다')
    def greet(self):
        self.wave() #자식 클래스는 알아서 인식함.
        # 부모의 method에 override했다.(덮어씀) 라고 표현

class Dog(Animal):
    def wag(self):
        print('꼬리를 흔든다')
    def greet(self):
        self.wag()
person = Human()
person.greet()

dog=Dog()
dog.greet()

cow=Cow()
cow.greet()


class Car():

    def run(self):
        print("차가 달립니다.")


class Truck(Car):

    def load(self):
        print("짐을 실었습니다.")

    # 이 아래에서 run 메소드를 오버라이드 하세요.
    def run(self):
        print("트럭이 달립니다.")


truck = Truck()
truck.run()