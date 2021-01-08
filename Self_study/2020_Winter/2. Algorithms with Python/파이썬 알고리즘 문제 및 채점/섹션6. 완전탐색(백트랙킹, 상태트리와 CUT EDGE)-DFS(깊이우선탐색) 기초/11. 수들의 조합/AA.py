import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


def DFS(L,ind):
    global cnt
    
    if L==k:
        sum=0
        for j in res:
            sum+=j
        if sum % m ==0 :
            cnt+=1
    else:
        for i in range(ind,n): # 같은 레벨에서 너비
            res[L]=x[i]
            DFS(L+1,i+1) # 계속되는 깊이

    

if __name__=='__main__':
    n,k=map(int,input().split())
    x=list(map(int,input().split()))
    m=int(input())
    res=[0]*k
    cnt=0
    DFS(0,0)
    print(cnt)
