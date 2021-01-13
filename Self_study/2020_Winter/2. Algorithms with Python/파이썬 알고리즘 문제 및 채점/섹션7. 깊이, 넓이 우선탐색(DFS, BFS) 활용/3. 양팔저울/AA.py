import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,sum):
    global res
    if L==n:
        if 0<sum<=s: # 음수는 대칭이기 때문에, 안봐도 상관없음. 0은 쓸모없음.
            res.add(sum) 
    else:
        DFS(L+1,sum+G[L]) # 왼쪽에 놓는다.
        DFS(L+1,sum-G[L]) # 오른쪽에 놓는다.
        DFS(L+1,sum) # 사용하지 않는다.
        

if __name__=='__main__':
    n=int(input())
    G=list(map(int,input().split()))
    s=sum(G)
    res=set() # 중복제거 목적으로 set자료구조 사용
    
    DFS(0,0)
    print(s-len(res))

