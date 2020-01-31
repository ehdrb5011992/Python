class Human():
    '''인간'''
    def __init__(self,name,weight):
        '''초기화 함수'''
        #라기 보단 초기값 설정함수 가 더 맞는듯. 말이 좀;

        self.name = name
        self.weight = weight #이렇게 디폴트를 정할 수 있음.

     def __str__(self): #그냥 그 자체로 함수임.
         '''문자열화 함수'''
         return("{} (몸무게 {}kg)".format(self.name,self.weight))
        #얘를 지정해 주므로써 이름을 갖게됨.
        #없으면 그냥 <__main__.Human object at 0x00000188237256D8>
        #이렇게 뜸
        #아래에서 donggyu 라는 새로운 변수에 휴먼클래스를 입력하고,
        #print(donggyu)라고 하면, 위가 출력됨.

    # def create(name, weight):
    #     person = Human()
    #     person.name = name
    #     person.weight = weight
    #     return person  #얘를 위에 __str__과 __init__을 사용해서 새롭게 만듬

    def eat(self):
        self.weight += 0.1
        print('{}가 먹어서 {}kg이 되었습니다.'.format(self.name, self.weight))

    def walk(self):
        self.weight -= 0.1
        print('{}가 걸어서 {}kg이 되었습니다.'.format(self.name, self.weight))


person = Human('사람',60.5) #저장한것 뿐인데 init함수가 실행됨.
#init함수는 인스턴스를 만드는 순간에 자동으로 호출되는 함수.
#얘로 내가만들었던 create함수를 대체할 수 있음.
print(person.name)
print(person.weight)
print(person)
person.eat()