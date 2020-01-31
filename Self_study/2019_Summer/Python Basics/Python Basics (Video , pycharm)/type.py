s='hello world'
type(s)
f=3.14
type(f)

i=42
type(i)
type(42.0)
42 == 42.0
isinstance(42,int) #자료형을 검사해주는 함수
isinstance(42.0,float)

my_list = [1, 2, 3]
my_dict = {"풀": 800, "색연필": 3000}
my_tuple = (1, 2, 3)
number = 10
real_number = 3.141592

print(type(my_list))
print(type(my_dict))
print(type(my_tuple))
print(type(number))
print(type(real_number))

type(5)
isinstance(5,float)
numbers1 = []
type(numbers1)
numbers2 = list(range(10))
numbers2
chracters = list("hello")
type(numbers2)
type(chracters)
isinstance(numbers1,list) #numbers1나 chracters같이 사용 가능한 list
numbers1 == list

#instance = 예)

list1 = [1, 2, 3]
list2 = [1, 2, 3]

if list1 is list1:
    print("당연히 list1과 list1은 같은 인스턴스입니다.")

if list1 == list2:
    print("list1과 list2의 값은 같습니다.")
    if list1 is list2:
        print("그리고 list1과 list2는 같은 인스턴스입니다.")
    else:
        print("하지만 list1과 list2는 다른 인스턴스입니다.")