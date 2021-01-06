import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

nece=input()
n=int(input())
cases=[]
for _ in range(n):
    cases.append(input())

def solution(cases,nece):
    from collections import deque
    for idx,val in enumerate(cases):
        dq=deque(nece)
        for x in val:
            if x in dq:
                if x != dq.popleft():
                    print('#{0:1d} NO'.format(idx+1))
                    break
        else:
            if len(dq) == 0:
                print('#{0:1d} YES'.format(idx+1))
            else:
                print('#{0:1d} NO'.format(idx+1))
solution(cases,nece)
