import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


def DFS(level,st): # 조합 꼭 숙지. 
    global cnt
    
    if level==m:
        for j in res:
            print(j,end=' ')
        print()
        cnt+=1
    else:
        for i in range(st,n+1): # 여기는 너비에 대한 영역
            if check[i]==0: 
                check[i]==1
                res[level]=i
                DFS(level+1,res[level]+1)
                check[i]==0

if __name__=='__main__':
    n,m=map(int,input().split())
    res=[0]*m
    check=[0]*(n+1)
    cnt=0

DFS(0,1)
print(cnt)
