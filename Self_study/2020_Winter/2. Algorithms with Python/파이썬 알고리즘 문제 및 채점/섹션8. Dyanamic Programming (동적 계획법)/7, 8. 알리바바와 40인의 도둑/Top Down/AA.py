import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

if __name__=='__main__':
    n=int(input())
    arr=[list(map(int,input().split())) for _ in range(n)]
    dp=[[0]*n for _ in range(n)]
    
    def DFS(y,x):
        if dp[y][x]>0:
            return dp[y][x]
        
        if x==0 and y==0:
            dp[y][x]=arr[y][x]
            return dp[y][x]
        elif x==0 :
            dp[y][x]=DFS(y-1,x)+arr[y][x]
            return dp[y][x]
        elif y==0 :
            dp[y][x]=DFS(y,x-1)+arr[y][x]
            return dp[y][x]
        else:
            dp[y][x]=min(DFS(y,x-1),DFS(y-1,x)) + arr[y][x]
            return dp[y][x]

    print(DFS(n-1,n-1))
