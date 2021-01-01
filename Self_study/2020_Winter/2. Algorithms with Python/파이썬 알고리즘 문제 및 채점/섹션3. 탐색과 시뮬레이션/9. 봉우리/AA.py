import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

N=int(input())
x=[list(map(int,input().split())) for _ in range(N)]

def padding(x,N):
    for i in range(N):
        x[i].insert(0,0)
        x[i].append(0)
    x.insert(0,[0]*(N+2))
    x.append([0]*(N+2))
    return x

def check_peak(x,N):
    cnt=0
    for i in range(1,(N+1)):
        for j in range(1,(N+1)):
            case1=x[i][j]>x[i-1][j]
            case2=x[i][j]>x[i+1][j]
            case3=x[i][j]>x[i][j-1]
            case4=x[i][j]>x[i][j+1]
            if case1 & case2 & case3 & case4 :
                cnt+=1
    return cnt

x=padding(x,N)
print(check_peak(x,N))
