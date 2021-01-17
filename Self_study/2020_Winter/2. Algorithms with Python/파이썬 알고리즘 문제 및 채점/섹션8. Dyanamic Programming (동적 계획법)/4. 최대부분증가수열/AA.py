import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


n=int(input())
arr=list(map(int,input().split()))
arr.insert(0,0) # 리스트의 1번 인덱스부터 시작하게 하려고 하나 미룸.
dy=[0]*(n+1)
dy[1]=1 # 직관적으로 알수 있어서 초기화
res=0

for i in range(2,n+1):
    max=0
    for j in range(i-1,0,-1):
        if arr[j] < arr[i] and dy[j] > max:
            max=dy[j]
    dy[i]=max+1
    if dy[i]>res:
        res=dy[i]
print(res)



