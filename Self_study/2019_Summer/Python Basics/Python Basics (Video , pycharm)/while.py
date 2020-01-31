selected = None
while selected not in ['가위','바위','보'] :
    selected = input('가위, 바위, 보 중에 선택하세요>')
print('선택된 값은',selected )


if selected not in ['가위','바위','보'] : #while은 if와 비슷하다.
    selected = input('가위, 바위, 보 중에 선택하세요>') #if는 1번, while은 여러번

patterns = ['가위','보','보']
for pattern in patterns:
    print(pattern)

length = len(patterns)
i = 0
while i < length : #while은 이렇게 조건으로. for문은 in과 늘 연동
    print(patterns[i])
    i=i+1
#while문이 쉬운경우, for문이 쉬운경우 각각이 있다.
#가위바위보를 출력하는 for문을 만드려면 복잡할것이다.


########################## while에 대해서 ################################## (간단예제)

selected = None #정의해줘야함.
while selected not in ['가위','바위','보'] :
    selected = input('가위, 바위, 보 중에 선택하세요>')

#for문은 정의하지 않아도 됨.
for i in range(int(1e+10)) :
    selected = input('가위, 바위, 보 중에 선택하세요>')
    if selected in ['가위', '바위', '보']:
        print('선택된 값은 %s' %selected)
        break

while True :
    selected = input('가위, 바위, 보 중에 선택하세요>')
    if selected in ['가위', '바위', '보']:
        break


patterns = ['가위', '바위', '보']
selected = None

for selected in patterns :
        selected = input('가위 바위 보 중에 선택하세요 : ')
        if selected not in patterns :
            print('가위 바위 보 중에만 선택해야 합니다')
        else :
            print(selected)

selected = None
rsp = ['가위', '바위', '보']

for i in range(len(rsp)):
    selected = input('가위, 바위, 보 중에 선택하세요>')
    if selected in rsp[i]:
        print('선택된 값은 {}'.format(selected))
        continue
    selected = input('가위, 바위, 보 중에 선택하세요')
    print(selected)

selected=None



