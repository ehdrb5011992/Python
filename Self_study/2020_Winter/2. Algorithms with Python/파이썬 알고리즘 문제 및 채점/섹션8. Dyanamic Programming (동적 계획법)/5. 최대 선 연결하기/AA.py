import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
num=list(map(int,input().split()))

num.insert(0,0)
dp=[0]*(n+1)
dp[1]=1 # 들어가는 수는, 선이 연결되는 개수.
res=-2147000000

for i in range(2,n+1):
    tmp=0
    for j in range(1,i):
        if num[j]<num[i] and dp[j]>tmp:
            tmp=dp[j]
    dp[i]=tmp+1

    if dp[i]>res:
        res=dp[i]
print(res)
