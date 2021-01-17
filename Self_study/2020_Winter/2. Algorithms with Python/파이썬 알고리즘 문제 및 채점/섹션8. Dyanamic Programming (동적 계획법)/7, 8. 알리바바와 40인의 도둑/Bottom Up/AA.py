import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
x=[list(map(int,input().split())) for _  in range(n)]

dp=[[0]*n for _ in range(n)]
dp[0][0]=3
i,j=0,0

for i in range(n): # y축        
    tmp=21470000000 
    for j in range(n): # x축
        if i==0 and j>=1:
            dp[i][j]=dp[i][j-1]+x[i][j]
        elif j==0 and i>=1:
            dp[i][j]=dp[i-1][j]+x[i][j]
        else:
            lx,ly= j-1,i
            ux,uy= j,i-1
            
            dp[i][j]=min(dp[uy][ux],dp[ly][lx])+x[i][j]
print(dp[n-1][n-1])
