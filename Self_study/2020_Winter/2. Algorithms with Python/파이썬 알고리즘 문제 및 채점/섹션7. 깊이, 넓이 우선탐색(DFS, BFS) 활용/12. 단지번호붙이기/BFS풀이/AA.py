import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


from collections import deque
dx=[-1,0,1,0]
dy=[0,1,0,-1]
cplx=[]

n=int(input())
mapp=[]
for _ in range(n):
    mapp.append(list(map(int,input())))

dq=deque()
for i in range(n):
    for j in range(n):
        if mapp[i][j]==1:
            mapp[i][j]=0
            cnt=1
            dq.append((i,j))
            while dq: # 와 훨씬 직관적이네
                now=dq.popleft()
                for x,y in zip(dx,dy):
                    xx=now[0]+x
                    yy=now[1]+y
                    if 0<=xx<n and 0<=yy<n and mapp[xx][yy]==1:
                        mapp[xx][yy]=0
                        dq.append((xx,yy))
                        cnt+=1
            cplx.append(cnt)

cplx.sort()
print(len(cplx))
for i in cplx:
    print(i)
