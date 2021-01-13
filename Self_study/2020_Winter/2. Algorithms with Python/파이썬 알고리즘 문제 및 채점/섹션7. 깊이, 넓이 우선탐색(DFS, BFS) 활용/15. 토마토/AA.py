import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

m,n=map(int,input().split())
box=[list(map(int,input().split())) for _ in range(n)]

dx=[-1,0,1,0]
dy=[0,1,0,-1]

from collections import deque
dq=deque()
day=0

for i in range(m):
    for j in range(n):
        if box[j][i]==1:
            dq.append((i,j))


while dq:
    st=len(dq)
    for _ in range(st):
        cx,cy=dq.popleft()
        for x,y in zip(dx,dy):
            xx=cx+x
            yy=cy+y
            if 0<=xx<m and 0<=yy<n:
                if box[yy][xx]==0:
                    box[yy][xx]=1
                    dq.append((xx,yy))
                
    day+=1
if any(box[i][j]==0 for i in range(n) for j in range(m)):
    print(-1)
else:
    print(day-1)
