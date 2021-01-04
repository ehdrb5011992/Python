import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


N,M=map(int,input().split())
x=list(map(int,input().split()))

sorted_x=sorted(x,reverse=True)

cnt=0
while sorted_x:
    hvy= sorted_x.pop(0)
    for pson in sorted_x:
        if hvy+pson <=M:
            sorted_x.remove(pson)
            cnt+=1
            break
    else:
        cnt+=1

print(cnt)
