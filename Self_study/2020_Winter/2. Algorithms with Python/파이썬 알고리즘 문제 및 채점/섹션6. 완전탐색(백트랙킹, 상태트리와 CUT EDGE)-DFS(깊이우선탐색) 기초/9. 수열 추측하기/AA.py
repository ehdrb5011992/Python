import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,sum):
    if L==n and sum==f: #종료 조건
        for x in p:
            print(x,end=' ') # 답출력
        sys.exit(0) # 첫번째껏만 보려고 자체를 종료

    else:
        for i in range(1,n+1): # 순열만듦. 작은것부터 보는게 사전순임.
            if ch[i]==0:
                ch[i]=1
                p[L]=i
                DFS(L+1,sum+(p[L]*b[L]))
                ch[i]=0


if __name__=="__main__":
    n,f = map(int,input().split())
    p=[0]*n # 1 4 
    b=[1]*n # 개수 공간. 맨 끝은 1로 초기화
    ch=[0]*(n+1)
    for i in range(1,n): # 개수를 만들자. 
        b[i] = b[i-1]*(n-i)// i
    DFS(0,0)
