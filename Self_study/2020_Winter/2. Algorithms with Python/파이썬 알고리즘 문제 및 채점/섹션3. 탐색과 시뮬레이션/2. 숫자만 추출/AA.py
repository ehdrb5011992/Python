import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

x= input()

def is_int(x):
    num=''
    for s in x:
        if s.isdigit():
            num += s
    return int(num)

def div_num(x):
    cnt=0
    for i in range(1,x+1):
        if x % i ==0:
            cnt+=1
    return cnt

answer = is_int(x)
print(answer)
print(div_num(answer))
