import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

from collections import deque
dq=deque()
dq.append((0,0))
dx=[-1,0,1,0]
dy=[0,1,0,-1]
n=7
board=[]
for i in range(n):
    board.append(list(map(int,input().split())))
res=[[0]*n for _ in range(n)] # 있으면 안뻗을꺼임.

while dq:
    
    x,y=dq.popleft()
    
    if (x,y) == (n-1,n-1):
        break
    
    for i,j in zip(dx,dy):
        next_x=x+i
        next_y=y+j
        if n-1>= next_x >=0 and n-1>= next_y >= 0 :
            if board[next_x][next_y]==0 and res[next_x][next_y] ==0 :
        
                dq.append((next_x,next_y))
                res[next_x][next_y]=res[x][y]+1
        
                    
        
if res[n-1][n-1]==0:
    print(-1)
else:
    print(res[n-1][n-1])
