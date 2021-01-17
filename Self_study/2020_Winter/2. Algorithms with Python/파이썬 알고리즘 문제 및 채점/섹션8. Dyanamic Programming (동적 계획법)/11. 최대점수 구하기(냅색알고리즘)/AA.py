import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n,m=map(int,input().split())
quest=[list(map(int,input().split())) for _ in range(n)]
dp=[0]*(m+1)

for score,time in quest:
    for j in range(m,time-1,-1):
        dp[j]=max(dp[j],dp[j-time]+score)

print(dp[m])
