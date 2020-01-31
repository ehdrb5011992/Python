for i in range(5) :
    print(i)
#range(5)  list는 아니지만 , 0,1,2,3,4 를 만들어냄

names = ['필수','영화','바둑이','귀도']
for i in range(len(names)) :
    name = names[i]
    print(' {}번 : {}'.format(i+1 , name))
for i,name in enumerate(names) :
    print(' {}번 : {}'.format(i+1 , name))

#'차'함수
print(chr(44032))
print(chr(44032+11172-1))

for i in range(11172) :
    print(chr(44032+i), end='')

days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for i, day in enumerate(days):
    #print('%d월의 날짜수는 %d일 입니다.'%(i+1,day))
    print('{}월의 날짜수는 {}일 입니다.'.format(i + 1, day))