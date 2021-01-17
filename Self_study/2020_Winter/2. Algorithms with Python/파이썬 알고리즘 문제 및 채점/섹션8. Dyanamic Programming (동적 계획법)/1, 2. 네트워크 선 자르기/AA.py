import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(stairs):
    if dp[stairs]>0:
        return dp[stairs]
    
    if stairs==1 or stairs==2:
        dp[stairs]=stairs
        return dp[stairs]
    else:
        dp[stairs]=DFS(stairs-1)+DFS(stairs-2)
        return dp[stairs]

if __name__ == '__main__':
    n=int(input())
    dp=[0]*(n+1)
    print(DFS(n))
