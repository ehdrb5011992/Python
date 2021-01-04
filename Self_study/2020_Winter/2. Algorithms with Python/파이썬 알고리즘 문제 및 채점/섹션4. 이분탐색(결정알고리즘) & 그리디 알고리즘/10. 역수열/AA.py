import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
a=list(map(int,input().split()))

seq=[0]*n # 그냥 n+1개 방 생성.
for i in range(n):
    for j in range(n):
        if a[i]==0 and seq[j]==0:
            seq[j]=i+1
            break
        elif seq[j]==0 :# 빈자리 발견
            a[i]-=1
for x in seq:
    print(x, end=' ')
    
