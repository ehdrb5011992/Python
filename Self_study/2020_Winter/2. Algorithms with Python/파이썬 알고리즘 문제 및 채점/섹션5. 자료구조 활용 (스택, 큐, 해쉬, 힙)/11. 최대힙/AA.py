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
            print(-hq.heappop(a)) # 음수를 받았으므로, 다시 음수를 바꿔 출력
    else:
        hq.heappush(a,-i) # 음수를 기준으로 정렬
