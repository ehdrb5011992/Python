import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=10
ladder=[list(map(int,input().split()))for _ in range(n)]

# 서 북 동 남
dx=[-1,0,1,0]
dy=[0,1,0,-1]

def DFS(x,y):
    global res
    if y==n-1:
        if ladder[y][x]==2:
            res=True
        else:
            res=False
            return
    else:
        lx,ly=x+dx[0],y+dy[0] # 서
        rx,ry=x+dx[2],y+dy[2] # 동
        dox,doy=x+dx[3],y-dy[3] # 남쪽. 이걸 주의해야함. 

        if 0<=lx<n and ladder[ly][lx]==1 and ch[ly][lx]==1:
            ch[ly][lx]=0
            DFS(lx,ly)
        elif 0<=rx<n and ladder[ry][rx]==1 and ch[ry][rx]==1:
            ch[ry][rx]=0
            DFS(rx,ry)
        elif ladder[doy][dox]!=0 and ch[doy][dox]==1:
            ch[doy][dox]=0
            DFS(dox,doy)

for st_ind in range(n):
    ch=[[1]*n for _ in range(n)]
    ch[0]=[0]*n
    if ladder[0][st_ind]==1:
        DFS(st_ind,0)
        if res:
            print(st_ind)
