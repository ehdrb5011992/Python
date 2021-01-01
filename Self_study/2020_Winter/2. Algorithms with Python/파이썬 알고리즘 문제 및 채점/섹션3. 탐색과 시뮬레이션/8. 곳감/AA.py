import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

N=int(input())
x=[list(map(int,input().split())) for _ in range(N)]
M=int(input())
y=[list(map(int,input().split())) for _ in range(M)]

def convert(x,y,N,M):
    for case in y:
        num_row,dire,dx=case[0]-1,case[1],case[2]
        idx = list(range(N))
        new_list = []
        if dire==0: #왼쪽
            new_idx = list(map(lambda x: (x+dx)% N ,idx))
            for i in new_idx:
                new_list.append(x[num_row][i])
            x[num_row] = new_list
        else: # 오른쪽
            new_idx = list(map(lambda x: (x-dx)% N ,idx))
            for i in new_idx:
                new_list.append(x[num_row][i])
            x[num_row] = new_list        
    return x

def summation(x,N):
    dx=ct=N//2
    cumsum=0
    for i in range(N):
        cumsum+=sum(x[i][(ct-dx):(ct+dx+1)])
        if i < N//2:
            dx-=1
        else:
            dx+=1
    return cumsum

x=convert(x,y,N,M)
print(summation(x,N))
