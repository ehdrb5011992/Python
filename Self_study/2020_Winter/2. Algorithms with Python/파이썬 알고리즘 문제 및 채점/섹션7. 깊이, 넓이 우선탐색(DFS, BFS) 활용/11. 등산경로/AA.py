import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(x,y):
    global cnt
    if (x,y) == (minmax_ind[0][0],minmax_ind[0][1]):
        cnt+=1
    else:
        for i,j in zip(dx,dy):
            xx=x+i
            yy=y+j
            if 0<=xx<=n-1 and 0<=yy<=n-1:
                if mapp[x][y]<mapp[xx][yy]:
                    DFS(xx,yy)



if __name__=="__main__":
    n=int(input())
    mapp=[]
    for _ in range(n):
        mapp.append(list(map(int,input().split())))
    
    tmp1=-2147000000
    tmp2=2147000000
    minmax_ind=[(n-1,n-1),(0,0)] # max / min

    for i in range(n):
        for j in range(n):
            if mapp[i][j] >tmp1:
                minmax_ind[0]=i,j
                tmp1 = mapp[i][j]
            
            if tmp2 > mapp[i][j]:
                minmax_ind[1]=i,j
                tmp2=mapp[i][j]
    
    cnt=0
    dx=[-1,0,1,0]
    dy=[0,1,0,-1]

    DFS(minmax_ind[1][0],minmax_ind[1][1])
    print(cnt)
