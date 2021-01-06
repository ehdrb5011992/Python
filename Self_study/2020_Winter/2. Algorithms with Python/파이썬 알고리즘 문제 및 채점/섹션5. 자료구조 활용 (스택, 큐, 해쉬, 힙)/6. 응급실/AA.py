import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


n,m = map(int,input().split())
x=list(map(int,input().split()))

def solution(x,m):
    from collections import deque
    dq=[(idx,val) for idx,val in enumerate(x)]
    dq=deque(dq)
    cnt=0
    while True:
        cur=dq.popleft()
        if any(cur[1]<x[1] for x in dq):
            dq.append(cur)
        else:
            cnt+=1
            if cur[0]==m:
                break

    return cnt

print(solution(x,m))
