import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

L=int(input())
a=list(map(int,input().split()))
m=int(input())

a.sort()
for _ in range(m):
    a[0]+=1
    a[L-1]-=1
    a.sort()
print(a[L-1]-a[0])
