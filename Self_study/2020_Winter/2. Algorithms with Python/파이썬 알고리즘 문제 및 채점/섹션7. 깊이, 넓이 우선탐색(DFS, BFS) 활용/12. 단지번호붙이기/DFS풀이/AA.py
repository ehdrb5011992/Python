import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

dx=[-1, 0, 1, 0]
dy=[0, 1, 0, -1]

def DFS(x,y):
    global cnt
    cnt+=1
    board[x][y]=0 # 방들어오면, 그 공간은 0으로
    for i in range(4):
        xx=x+dx[i]
        yy=y+dy[i]
        if 0<=xx<n and 0<=yy<n and board[xx][yy]==1:
            DFS(xx,yy)


if __name__=="__main__":
    n=int(input())
    board=[list(map(int,input())) for _ in range(n)]
    res=[]
    for i in range(n):
        for j in range(n):
            if board[i][j]==1:
                cnt=0 # 단지가 나타날때마다 초기화
                DFS(i,j) # 1 하나 발견하면, DFS시작. (관련된거 다 잡아먹음)
                res.append(cnt)
res.sort()
print(len(res))
for i in res:
    print(i)

