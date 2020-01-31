class Animal(): #부모클래스

    def walk(self):
        print('걷는다')
    def eat(self):
        print('먹는다')
    def greet(self):
        print("인사한다")

class Human(Animal):
    def wave(self):
        print('손을 흔들면서')
    def greet(self):
        self.wave()
        super().greet() #자식클래스에 override한 매소드에서
                        # 추가하고 싶은경우는 옆처럼 하면 됨.
                        # 얘는 __init__에서 자주씀
                        # super()는 부모의 매서드를 불러오는 방법.
                        #자세히보면 super은 print와 같은 파란색

person = Human()
person.greet()


class Animal():  # 부모클래스
    def __init__(self,name):
        self.name = name
    def walk(self):
        print('걷는다')
    def eat(self):
        print('먹는다')
    def greet(self):
        print("{}이/가 인사한다".format(self.name))

class Human(Animal):

    def __init__(self,name,hand): #변수 누적.
        super().__init__(name) #부모가 처리해야할 값 초기화
                               #이게 복잡할 때 유용하다.
                               #이때 __init__은 Animal의 init이고,
                               #name은 변수.

        self.hand = hand #내가 추가하고 싶은 값 초기화

    def wave(self):
        print('{}을 흔들면서'.format(self.hand))
    def greet(self):
        self.wave() # <-- wave임.
        super().greet() # <-- super()은 부모클래스.

person = Human("사람",'오른손')
person
