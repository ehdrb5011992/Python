import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
x=[list(map(int,input().split())) for _ in range(n)]

def apple_tree(x,N):

    dx=0
    tot=0
    st= N//2
    for i in range(N):
        tot += sum(x[i][(st-dx):(st+dx+1)])
        if i < (N-1)//2:
            dx+=1
        else:
            dx-=1
    return(tot)
print(apple_tree(x,n))
