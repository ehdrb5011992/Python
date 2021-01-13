import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

from collections import deque

max=10000
ch=[0]*(max+1) #좌표에 따라 인덱스를 그대로 사용하기 위해.
dis=[0]*(max+1)
n,m=map(int,input().split())

ch[n]=1
dis[n]=0
dQ=deque()
dQ.append(n)
while dQ: # BFS는 queue가 비어있으면 종료하는거임.
    now=dQ.popleft()

    if now==m:  #여기가 종료조건
        break


    # 여기를 돌리는거임.
    for next in (now-1,now+1,now+5): # 이렇게 현재 값에서 세 가닥으로 뽑는다는거임.
        if 0<next<=max:
            if ch[next]==0:
                dQ.append(next)
                ch[next]=1
                dis[next]=dis[now]+1

print(dis[m])
