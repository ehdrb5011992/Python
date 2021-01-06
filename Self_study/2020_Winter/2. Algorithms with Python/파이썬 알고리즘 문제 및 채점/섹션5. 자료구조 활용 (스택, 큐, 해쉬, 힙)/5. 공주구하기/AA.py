import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


n,k=map(int,input().split())
x=list(range(1,n+1))

def solution(x,k):
    while len(x)>1:
        for i in range(k-1):
            x.append(x.pop(0))
        x.pop(0)
    return x[0]        
print(solution(x,k))
