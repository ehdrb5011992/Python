list=[1,2,3,4,5]

for i,v in enumerate(list) : #인덱스와 값을 튜플로 리턴해줌.
    print('{}번째 값: {}'.format(i,v))

for a in enumerate(list) : #인덱스와 값을 튜플로 리턴해줌.
    print('{}번째 값: {}'.format(a[0],a[1]))

for a in enumerate(list) : #인덱스와 값을 튜플로 리턴해줌.
    print('{}번째 값: {}'.format(*a)) #*a는 튜플 a를 쪼개라 라는 의미.

ages={'tod':35,'jane':23,'paul':62}
for key,value in ages.items() :
    print('{}의 나이는:{}'.format(key,value))
for a in ages.items() :
    print('{}의 나이는:{}'.format(*a))


