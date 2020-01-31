#9
print("first",end="");print("second")
#15
t1='python';t2='java';print((t1+" "+t2+' ')*4)
#22
license_plate = "24가 2210" ; print(license_plate[-4:])
#23
string='홀짝홀짝홀짝' ; print(string[::2])
print(''.join([x for x in string if x == "홀" ]))
#24
string = "PYTHON" ; print(string[::-1])
#25
phone_number = '010-1111-2222' ; print(phone_number.replace('-',' '))
#28 문자열은 변경이 불가능함.
lang='python' ; lang[0] = 'P' ; print(lang) # Wrong
"P"+lang[1:] # Correct
#30
string = 'abcd' ; string.replace('b','aa') ; print(string)
#53
nums=[i for i in range(1,11)] ; print(nums[1::2])
#56
interest=['삼성전자','LG전자','Naver','SK하이닉스','미래에셋대우']
print(" ".join(interest))
#61 복사가 어떻게 일어나는지 61과 62를 통해 비교해서 알아놓기.
interest_0 = ['삼성전자','LG전자','SK Hynix']
interest_1 = interest_0
interest_1[0] = "Naver"
print(interest_0) ; print(interest_1)
#62
interest_0 = ['삼성전자','LG전자','SK Hynix']
interest_1 = interest_0[:2]
interest_1[0] = "Naver"
print(interest_0) ; print(interest_1)
#67
t=('a','b','c') ;  t=(t[0].upper(),) + t[1:3] ; print(t)
#71
a,b,*c = (0,1,2,3,4,5) ; a ; b ;c
#72
scores = [8.8, 8.9, 8.7, 9.2, 9.3, 9.7, 9.9, 9.5, 7.8, 9.4]
_,_,*valid_score =scores
#75
icecream_price = {'메로나':1000, '폴라포':1200, '빵빠레':1800}
icecream_price['메로나'] = 1200
icecream_price['월드콘'] = 1500
#88
new_prodcut = {'팥빙수':2700}
icecream_price.update(new_prodcut) ; print(icecream_price)
#101
hi = input()
print(hi*2) #이렇게 코딩하면 받고 출력까지 한번에수행.
#104
x = int(input("입력값: "))
print( "출력값: ",min( x+20,255) ) #범위를 가를땐, min , max이용.
#106
x=input("현재시간 : ")
if x.split(':')[1] != '00' :
    print("정각이 아닙니다.")
else:
    print("정각 입니다.")
#108
warn_investment_list=['Microsoft','Google','Naver','Kakao','SAMSUNG','LG']
warn_investment_list
x=input("투자 종목을 입력해주세요: ")
if x.lower() in [y.lower() for y in warn_investment_list] :
    print("투자 경고 종목입니다.")
else:
    print("투자 경고 종목이 아닙니다.")
# 113
x,y = input('입력 :').split(' ')
if y == '달러' :
    print(int(x)*1167,'원')
elif y == '엔' :
    print(int(x)*1.096,'원')
elif y == '유로' :
    print(int(x)*1268,'원')
elif y == '위안' :
    print(int(x)*171,'원')
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
#117
x = int(input('주민등록번호: ').split('-')[1][0])
if x in (1,3) :
    print('남자')
elif x in (2,4) :
    print('여자')
else :
    print('??')
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
#138
my_list = [3, 1, 7, 10, 5, 6]
for i in my_list :
    if 5 <i < 10 :
        print(i)
#또는 and. 그러나 &는 안됨. 파이썬에선 and와 &가 다름. (R은 같음)
# x and y에서 x가 False면 x반환, x가 True면 y반환 // &는 비트연산자 <- 최대한거르는목적
# x or y 에서 x가 True면 x반환, x가 False면 y반환 // |도 비트연산자 <- 최대한포함목적
for i in my_list :
    if i>5 and i < 10 :
        print(i)
#143
my_list = ["A", "b", "c", "D"]
for i in my_list:
    if i.isupper():
        print(i.lower(),end='')
    else :
        print(i.upper(),end='')
#150
my_list = [3, 4, 5]
a=0
for i in my_list:
    a += i
print(a)
