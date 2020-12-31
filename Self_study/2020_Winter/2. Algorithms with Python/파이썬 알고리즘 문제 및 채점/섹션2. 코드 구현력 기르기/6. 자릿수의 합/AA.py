import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
a = list(map(int,input().split()))

def digit_sum(x):
    sum = 0
    for i in str(x):
        sum += int(i)

    return sum


max = -2147000000
# 2의 31제곱인데, C나 C++로 넘어가게 된다면 4바이트를 지원하기때문.
# 정확한 값은 아닌데, 뒤의 6자리는 외우기 까다로우니 그냥 0으로 대체

for x in a:
    tot=digit_sum(x)
    if tot > max:
        max = tot
        res = x  # 그때의 값을 res로 저장. 굳이 index를 만들필요가 없음.
print(res)



