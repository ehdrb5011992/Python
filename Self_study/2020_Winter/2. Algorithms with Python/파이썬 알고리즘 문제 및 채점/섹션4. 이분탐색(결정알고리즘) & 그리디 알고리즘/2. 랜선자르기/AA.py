import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

k,n=map(int,input().split())
x=[]
for _ in range(k):
    x.append(int(input()))


def line_split(x,N):
    lt=1
    rt=214700000 # 어차피 탐색할거기 때문에, 4개의 사례중 가장 큰값보다도 더 큰범위면됨.
    ans=-2147000000

    while lt<=rt:
        mid=(lt+rt)//2

        tot=0
        for i in x:
            tot+=i//mid

        if tot >= N:
            ans = mid
            lt = mid+1
        else:
            rt = mid-1
    return ans

print(line_split(x,n))

