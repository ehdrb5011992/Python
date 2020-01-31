#먹고 손을 흔들고 걷는다(class의 확장)

class Animal():
    ''' 동물 '''

class Human():
    def walk(self):
        print('걷는다')
    def eat(self):
        print('먹는다')
    def wave(self):
        print('손을 흔든다')

class Dog():
    def walk(self):
        print('걷는다')
    def eat(self):
        print('먹는다')
    def wag(self):
        print('꼬리를 흔든다')

person = Human()
person.walk()
person.eat()
person.wave()

dog=Dog()
dog.walk()
dog.eat()
dog.wag()

#코드가 낭비된다. (같은코드를 썻기때문.)
#사람과 개는 다른개념이지만, 많은것을 공유함.
#이런걸 상속으로 해결. 아래처럼.

class Animal(): #부모클래스
    def walk(self):
        print('걷는다')
    def eat(self):
        print('먹는다')

class Human(Animal): #상속의 개념. Human클래스는 자식클래스
    def wave(self):
        print('손을 흔든다')

class Dog(Animal):
    def wag(self):
        print('꼬리를 흔든다')

person = Human()
person.walk()
person.eat()
person.wave()

dog=Dog()
dog.walk()
dog.eat()
dog.wag()