import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
x=list(map(int,input().split()))
m=int(input())


dp=[2147000000]*(m+1)
dp[0]=0
for i in x:
    for j in range(i,m+1):
        dp[j]=min(dp[j],dp[j-i]+1)

print(dp[m])
