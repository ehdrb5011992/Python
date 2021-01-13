import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

from collections import deque
n=int(input())
city=[list(map(int,input().split())) for _ in range(n)]


res=-2147000000
dx=[-1,0,1,0]
dy=[0,1,0,-1]
dq=deque()


rain_max=-2147000000
for i in range(n):
    for j in range(n):
        if city[i][j]> rain_max:
            rain_max=city[i][j]


for rain in range(1,rain_max):
    
    cnt=0
    ch=[[1]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if city[i][j] <= rain:
                ch[i][j]=0 # 0이면 침수.

    for i in range(n):
        for j in range(n):
            if ch[i][j]==1: # 땅이있음.
                cnt+=1
                ch[i][j]=0 # 땅을 없앰
                dq.append((i,j)) # 인덱스저장
                while dq:
                    cx,cy=dq.popleft() # 출발
                    for x,y in zip(dx,dy):
                        xx=cx+x
                        yy=cy+y
                        if 0<=xx<n and 0<=yy<n and ch[xx][yy]==1:
                            ch[xx][yy]=0
                            dq.append((xx,yy))
    else:        
        if cnt>res:
            res=cnt
else:
    print(res)
