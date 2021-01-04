import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

N,M = map(int,input().split())
x=list(map(int,input().split()))

x_sort=sorted(x)

s=0
e=N-1
while True:
    mid=(s+e)//2
    if x_sort[s] == M:
        mid=s
        break
    elif x_sort[e]==M:
        mid=e
        break
    elif x_sort[mid]==M:
        break
    
    if M>x_sort[mid]:
        s=mid
    elif M<x_sort[mid]:
        e=mid
    else:
        break
print(mid+1)
