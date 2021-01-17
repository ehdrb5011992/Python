import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
dy=[0]*(n+1)
dy[1]=1
dy[2]=2
for i in range(3,n+1):
    dy[i]=dy[i-1]+dy[i-2]

print(dy[n])
