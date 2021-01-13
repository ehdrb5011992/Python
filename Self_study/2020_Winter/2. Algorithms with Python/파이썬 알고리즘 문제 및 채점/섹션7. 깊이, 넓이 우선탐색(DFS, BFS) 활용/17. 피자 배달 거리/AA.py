import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,s):
    global res
    if L==m:
        sum=0
        for j in range(len(hs)):
            x1,y1=hs[j][0],hs[j][1]
            dis=2147000000
            for x in cb:
                x2,y2=pz[x][0],pz[x][1]
                dis=min(dis,abs(x1-x2)+abs(y1-y2)) # 작은값이 갱신
            sum+=dis
        if sum<res:
            res=sum

            
    else:
        for i in range(s,len(pz)):
            cb[L]=i # 피자집의 번호
            DFS(L+1,i+1)
    
    
    



if __name__=='__main__':
    n,m=map(int,input().split())
    board=[list(map(int,input().split())) for _ in range(n)]
    hs=[]
    pz=[]
    cb=[0]*m # 선택되는 피자
    res=2147000000
    for i in range(n):
        for j in range(n):
            if board[i][j]==1:
                hs.append((i,j))
            elif board[i][j]==2:
                pz.append((i,j))
    DFS(0,0)
    print(res)

