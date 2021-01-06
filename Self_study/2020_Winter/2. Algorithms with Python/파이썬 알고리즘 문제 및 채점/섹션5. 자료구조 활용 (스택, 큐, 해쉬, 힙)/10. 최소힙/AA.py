import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

import heapq as hq

a=[]
n=[]
while True:
    num=int(input())
    n.append(num)
    if num==-1:
        break
    
for i in n:
    if i==-1:
        break
    elif i==0:
        if len(a)==0:
            print(-1)
        else:
            print(hq.heappop(a))
    else:
        hq.heappush(a,i)
