import sys
#sys.stdin=open("input.txt",'rt') 
n, k=map(int,input().split()) # 하나씩 차례차례 매핑함.

cnt=0
for i in range(1,n+1):
    if n%i==0:
        cnt+=1
    if cnt==k:
        print(i)
        break
else:
    print(-1)
    
