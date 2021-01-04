import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

N,C = map(int,input().split())
x=[]
for _ in range(N):
    x.append(int(input()))
x.sort()

def horse(mid,x):
    cnt=0
    p1=x[0]
    for p2 in x:
        dist=p2-p1
        if dist >= mid:
            cnt+=1
            p2=p1            
    
    return cnt

def solution(x,C):
    lt=x[0]
    rt=x[-1]
    res=0
    while lt<=rt: # mid와 rt, lt는 전부 말과 말(두마리) 사이의 거리.
        mid = (lt+rt) // 2
        if horse(mid,x) >= C:
            res=mid
            lt = mid+1
        else:
            rt = mid-1
    return res

print(solution(x,C))
