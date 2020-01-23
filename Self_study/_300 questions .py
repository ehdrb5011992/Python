#1.
print("Hello World")
#2.
print("Mary's cosmetics")
#3
print('신씨가 소리질렀다. "도둑이야".')
#4
print("C:\Windows")
#5
print("안녕하세요. \n만나서\t\t반갑습니다.")
#6.
print("오늘은","일요일")
#7
print("naver",'kakao','sk','samsung',sep=';')
#8
print("naver",'kakao','sk','samsung',sep='/')
#9
print("first",end="");print("second")
#10
string = "dk2jd923i1jdk2jd93jfd92jd918943jfd8923"
print(len(string))
#11
a='3';b='4';print(a+b)
#12
s='hello';t='python';print(s+"!",t)
#13
print("hi"*3)
#14
print('-'*80)
#15
t1='python';t2='java';print((t1+" "+t2+' ')*4)
#16
print(20000*10)
#17
2+2*3
#18
a=128;print(type(a));a='132';print(type(a))
#19
num_str = '720';print(int(num_str))
#20
num=100 ; print(str(num))
#21
lang = 'python';print(lang[0],lang[2])
#22
license_plate = "24가 2210" ; print(license_plate[-4:])
#23
string='홀짝홀짝홀짝' ; print(string[::2])
print(''.join([x for x in string if x == "홀" ]))
#24
string = "PYTHON" ; print(string[::-1])
#25
phone_number = '010-1111-2222' ; print(phone_number.replace('-',' '))
#26
print(phone_number.replace('-',''))
#27
url = 'http://sharebook.kr' ; print(url[-2:])
#28
lang='python' ; lang[0] = 'P' ; print(lang)
#29
string = 'abcdfe2a354a32a' ; print(string.replace('a',"A"))
#30
string = 'abcd' ; string.replace('b','B') ; print(string)
#41
movie_rank = ['닥터스트레인지','스플릿','럭키']
#42
movie_rank.append('배트맨') ; movie_rank
#43
movie_rank.insert(1,"슈퍼맨") ; movie_rank
#44
movie_rank.remove('럭키') ; movie_rank
#45
del movie_rank[2:] ; movie_rank
#46
lang1 = ['c','c++','java'] ; lang2=['python','go','c#'] ; print(lang1+lang2)
#47
nums = [1,2,3,4,5,6,7] ; print('max:', max(nums)) ; print('min:',min(nums))
#48
nums = [1,2,3,4,5] ; print(sum(nums))
#49
cook=['피자','김밥','등등해서','김치전'];print(len(cook))
#50
nums=[1,2,3,4,5] ; print(sum(nums)/len(nums))
#51
price = ['20180728',100,130,140,150,160,170];print(price[1:7])
#52
nums=[1,2,3,4,5,6,7,8,9,10] ; print(nums[::2])
#53
nums=[i for i in range(1,11)] ; print(nums[1::2])
#54
nums=[i for i in range(1,6)] ; print(nums[::-1])
#55
interest=['삼성전자','LG전자','Naver'];print(interest[0],interest[2])
#56
interest=['삼성전자','LG전자','Naver','SK하이닉스','미래에셋대우']
print(" ".join(interest))
#57
interest=['삼성전자','LG전자','Naver','SK하이닉스','미래에셋대우']
print("/".join(interest))
#58
interest=['삼성전자','LG전자','Naver','SK하이닉스','미래에셋대우']
print("\n".join(interest))
#59
string='삼성전자/LG전자/Naver' ; interest=string.split('/') ; print(interest)
#60
string="삼성전자/LG전자/Naver/SK하이닉스/미래에셋대우"
interest=string.split('/')
print(interest)
#61
interest_0 = ['삼성전자','LG전자','SK Hynix']
interest_1 = interest_0
interest_1[0] = "Naver"
print(interest_0) ; print(interest_1)
#62
interest_0 = ['삼성전자','LG전자','SK Hynix']
interest_1 = interest_0[:2]
interest_1[0] = "Naver"
print(interest_0) ; print(interest_1)
#63
my_variable=()
#64
t = (1, 2, 3) ;  t[0] = 'a'
#65
tup = (1,)
#66
t=1,2,3,4 ; type(t)
#67
t=('a','b','c') ;  t=(t[0].upper(),) + t[1:3] ; print(t)
#68
interest = ('삼성전자','LG전자','SK Hynix') ; list(interest)
#69
interest = ['삼성전자','LG전자','SK Hynix'] ; tuple(interest)
#70
my_tuple = (1, 2, 3) ; a, b, c = my_tuple ; print(a + b + c)
#71
a,b,*c = (0,1,2,3,4,5) ; a ; b ;c
#72
scores = [8.8, 8.9, 8.7, 9.2, 9.3, 9.7, 9.9, 9.5, 7.8, 9.4]
_,_,*valid_score=scores
#이하 90번까지는 했으므로 생략.
#92
print(3==5)
#93
print(3<5)
#94
x=4 ; print(1<x<5)
#95
print((3==3) and (4!=3))
#96
print(3 >= 4) #print(3 => 4)
#97
if 4<3 :
    print("Hello World")
#98
if 4 < 3:
    print("Hello World.")
else:
    print("Hi, there.")
#99
if True :
    print ("1")
    print ("2")
else :
    print("3")
print("4")
#100
if True :
    if False:
        print("1")
        print("2")
    else:
        print("3")
else :
    print("4")
print("5")
#101
hi = input()
print(hi*2) #이렇게 코딩하면 받고 출력까지 한번에수행.
#102
x = input("숫자를 입력하세요: ")
print(eval(x)+10)
#103
x = int(input("숫자를 입력하세요: "))
if x % 2 == 0 :
    print("짝수")
else:
    print("홀수")
#104
x = int(input("입력값: "))
if 0< x <= 235 :
    print(x+20)
elif x > 235:
    print(255)
x = int(input("입력값: "))
print( "출력값: ",min( x+20,255) )
#105
x = int(input("입력값: "))
print( "출력값: ",min( x-20,0) )
#106
x=input("현재시간 : ")
if x.split(':')[1] != '00' :
    print("정각이 아닙니다.")
else:
    print("정각 입니다.")
#107
fruit = ['사과','포도','홍시']
x = input("좋아하는 과일은? ")
if x  in fruit :
    print("정답입니다.")
else:
    print("오답입니다.")
#108
warn_investment_list=['Microsoft','Google','Naver','Kakao','SAMSUNG','LG']
warn_investment_list
x=input("투자 종목을 입력해주세요: ")
if x.lower() in [y.lower() for y in warn_investment_list] :
    print("투자 경고 종목입니다.")
else:
    print("투자 경고 종목이 아닙니다.")
#109
fruit = {"봄":'딸기','여름':'토마토','가을':'사과'}
x=input("제가 좋아하는 계절은: ")
if x in fruit.keys():
    print("정답입니다.")
else:
    print("오답입니다.")
#110
fruit = {"봄":'딸기','여름':'토마토','가을':'사과'}
x=input("좋아하는 과일은?: ")
if x in fruit.values():
    print("정답입니다.")
else:
    print("오답입니다.")
#111
x=input("문자를 입력하시오: ")
if x.islower()  :
    print(True)
else:
    print(False)
#112
score = int(input("score:"))
if score > 80 :
    print("grade is A")
elif score > 60 :
    print("grade is B")
elif score > 40 :
    print("grade is C")
elif score > 20 :
    print("grade is D")
else :
    print("grade is E")
#113
x,y = input('입력 :').split(' ')
if y == '달러' :
    print(int(x)*1167,'원')
elif y == '엔' :
    print(int(x)*1.096,'원')
elif y == '유로' :
    print(int(x)*1268,'원')
elif y == '위안' :
    print(int(x)*171,'원')
#114
x = int(input('input number1: '))
y = int(input('input number2: '))
z = int(input('input number3: '))
print(max(x,y,z))
#115
x = input('휴대전화 번호 입력: ').split('-')
y = {'011':'SKT' , '016':'KT' , '019':'LGU', '010':'알수없음'}
print('당신은 {} 사용자입니다.'.format(y[x[0]]))
#116
x = input('우편번호: ')[2]
y = { '강북구' : ['0','1','2'] , '도봉구':['3','4','5'] ,'노원구' : ['6','7','8','9'] }
def reverse(x,y):
    for a in y:
        if x in y[a] :
            return(a)
    raise ValueError("숫자 입력하세요.")
# 리스트 내의 리스트 풀기 list_of_lists = sum( y.values() ,[])
#118은 비슷하므로 생략
#119 <- numpy를 배우면 쉽게 요소별곱을 시행할 수 있음.
import numpy as np
x = input('주민등록번호: ').split('-')
num = list("".join(x))
num2=list(map(int, num))
a1 = np.array(num2)[:-1]
a2 = np.array([i for i in range(2,10)] + [j for j in range(2,6)])
first = sum(a1*a2) % 11 ; second = 11-first
if second == int(num[-1]) :
    print("유효한 주민등록번호입니다.")
else:
    print("유효하지 않은 주민등록번호입니다.")
#120 생략
#121
for i in ['가','나','다','라'] :
    print(i)
#122
for 변수 in ["사과", "귤", "수박"]:
    print(변수)
#123
for 변수 in ["사과", "귤", "수박"]:
    print(변수)
    print("--")
#124
for 변수 in ["사과", "귤", "수박"]:
    print(변수)
print("--")
#125
menu = ["김밥", "라면", "튀김"]
for i in menu:
    print('오늘의 메뉴:', i)
#126 생략
#127
pets = ['dog', 'cat', 'parrot', 'squirrel', 'goldfish']
for i in pets:
    print(i,len(i))
#128
prices = [100, 200, 300]
for i in prices :
    print(i + 10)
#129
prices = ["129,300", "1,000", "2,300"]
for i in prices :
    print(int(i.replace(",","")))
#130
menu = ["면라", "밥김", "김튀"]
for i in menu :
    print(i[::-1])
#131
my_list = ["가", "나", "다", "라"]
for i in my_list[1:]:
    print(i)
#132
my_list = [1, 2, 3, 4, 5, 6]
for i in my_list[::2]:
    print(i)
#133
my_list = [1, 2, 3, 4, 5, 6]
for i in my_list[1::2]:
    print(i)
#134
my_list = ["가", "나", "다", "라"]
for i in my_list[::-1]:
    print(i)
#135
my_list = [3, -20, -3, 44]
for i in my_list :
    if i < 0:
        print(i)
#136
my_list = [3, 100, 23, 44]
for i in my_list:
    if i % 3 == 0 :
        print(i)
#137
my_list = ["I", "study", "python", "language", "!"]
for i in my_list:
    if len(i) >= 3 :
        print(i)
#138
my_list = [3, 1, 7, 10, 5, 6]
for i in my_list :
    if 5 <i < 10 :
        print(i)
for i in my_list :
    if i>5 and i < 10 :
        print(i)
#139
my_list = [13, 21, 12, 14, 30, 18]
for i in my_list:
    if i > 10 and i < 20 and i % 3 == 0 :
        print(i)
#140
my_list = [3, 1, 7, 12, 5, 16]
for i in my_list:
    if i % 3 ==0 or i % 4 == 0 :
        print(i)
#141
my_list = ["A", "b", "c", "D"]
for i in my_list:
    if i.isupper() :
        print(i)
#142
my_list = ["A", "b", "c", "D"]
for i in my_list:
    if i.islower() :
        print(i)
#143
my_list = ["A", "b", "c", "D"]
for i in my_list:
    if i.isupper():
        print(i.lower(),end='')
    else :
        print(i.upper(),end='')
#144
file_list = ['hello.py', 'ex01.py', 'ch02.py', 'intro.hwp']
for i in file_list:
    print(i.split('.')[0])
#145
filenames = ['intra.h', 'intra.c', 'define.h', 'run.py']
for i in filenames:
    if i.split('.')[1] == 'h' :
        print(i)
#146
filenames = ['intra.h', 'intra.c', 'define.h', 'run.py']
for i in filenames:
    if i.split('.')[1] == 'h' or i.split('.')[1] == 'c':
        print(i)
#147
my_list = [3, -20, -3, 44]
new_list=[]
for i in my_list:
    if i >0 :
        new_list.append(i)
print(new_list)
#148
my_list = ['A', "b", "c", "D"]
upper_list=[]
for i in my_list :
    if i.isupper():
        upper_list.append(i)
print(upper_list)
#149
my_list = [3, 4, 4, 5, 6, 6]
sole_list=[]
for i in my_list:
    if i not in sole_list:
        sole_list.append(i)
print(sole_list)
#150
my_list = [3, 4, 5]
a=0
for i in my_list:
    a += i
print(a)