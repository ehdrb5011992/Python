#index , insert , extend, sort , reverse
#split,join(문자열인 경우)
#list1[15:5:-3] 스텝 -3
#numbers[1:3] = [77,88,99,100,200,500] #같은 개수가 아니어도 가능
#반대도 성립

list1= [135,462,27,2753,234]
list1.index(27)
list1.index(50)

if 50 in list1 :
    list1.index(50)

list2 = [1,2,3] + [4,5,6]
list1
a = list1.extend([9,10,11])
list1

list1.insert(2,999)
list1.insert(10000,555)
list1.sort()
list1.reverse()
###############################################
my_list = [1,2,3,4,5,6]
my_list[0]
my_list[1]
str= "hello world"
str[0]
str[1]

3 in my_list
9 in my_list
"h" in str
"z" in str
my_list.index(5)
str.index("r")
characters = list('abcdef')
characters
words = "hello world는 프로그래밍을 배우기 위해 아주 좋은 사이트입니다."
word_list = words.split()
word_list
time_str = "10:35:27"
time_list = time_str.split(":")
time_list
"".join(word_list)
###############################################

list = [1,2,3,4,5]
list[1]
text = "hello world"
text[1]
text[1:5] #공백 전까지 가져와라
text[1]
text[5] #공백임.
list=['영','일','이','삼','사','오']
list[1:3]

list[0:2]
list[2:len(list)]
list[2:]
list[:2]
list[:] #새로운 것을 만들어서 넘겨주는 행위 (복사)
list1=[31,4632,7235,23,61]
list2=list1[:]
list1
list2

list1.sort()
list1
list2
###############################################

rainbow = ["빨", "주", "노", "초", "파", "남", "보"]
rainbow[-3:]
###############################################
#지우고 시작
list1 = list(range(20))
list1
list1[5:15]
list1[5:15:2]
list1[5:15:3]
list1[15:5:-3]
list1[::3] #0부터 3개만큼 띄워서 값을 가져옴.
list1[::-3] #거꾸로 가져옴. 스트링도 가능함.
###############################################

numbers = [0,1,2,3,4,5,6,7,8,9]
numbers = list(range(10))
del numbers[0]
numbers
numbers[:5]
del numbers[:5] #영역 지우기도 가능.
numbers[1:3]
numbers[1:3] = [77,88] #한번에 바꾸기도 가능
numbers[1:3] = [77,88,99,100,200,500] #같은 개수가 아니어도 가능
numbers[1:4] = [8]
numbers