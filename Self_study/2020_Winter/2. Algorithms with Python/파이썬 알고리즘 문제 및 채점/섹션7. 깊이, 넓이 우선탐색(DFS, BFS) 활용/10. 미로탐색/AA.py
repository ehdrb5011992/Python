import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

dx=[-1,0,1,0]
dy=[0,1,0,-1]
n=7
cnt=0
res=[]
board=[]
for _ in range(n):
    board.append(list(map(int,input().split())))
board[0][0]=1 # 초기화

def DFS(x,y):
    global cnt
    if (x,y)==(6,6):
        cnt+=1
    else:
        for i,j in zip(dx,dy):
            next_x = x+i
            next_y = y+j
            if 0<=next_x <= (n-1) and 0<=next_y<=(n-1):
                if board[next_x][next_y]==0:
                    board[next_x][next_y]=1
                    DFS(next_x,next_y)
                    board[next_x][next_y]=0

DFS(0,0)
print(cnt)
