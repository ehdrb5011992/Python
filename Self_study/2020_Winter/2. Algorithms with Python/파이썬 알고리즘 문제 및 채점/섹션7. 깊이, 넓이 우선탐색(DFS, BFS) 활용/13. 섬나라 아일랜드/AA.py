import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
board=[]
for _ in range(n):
    board.append(list(map(int,input().split())))

from collections import deque
dx=[-1,-1,0,1,1,1,0,-1]
dy=[0,1,1,1,0,-1,-1,-1]

dq=deque()
res=0

for i in range(n):
    for j in range(n):
        if board[i][j]==1:
            board[i][j]=0
            res+=1
            dq.append((i,j))
            while dq:
                cx,cy=dq.popleft()
                for x,y in zip(dx,dy):
                    xx=cx+x
                    yy=cy+y
                    if 0<=xx<n and 0<=yy<n: 
                        if board[xx][yy]==1:
                            board[xx][yy]=0
                            dq.append((xx,yy))

print(res)
