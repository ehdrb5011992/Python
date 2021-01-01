import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

a=list(range(0,21))
for _ in range(10):
    s,e = map(int,input().split())
    for i in range((e-s+1)//2):
        a[s+i],a[e-i]=a[e-i],a[s+i]
a.pop(0)
for i in a:
    print(i,end=' ')
