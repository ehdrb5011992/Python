import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
x=[]
for i in range(n):
    a = list(map(int,input().split()))
    x.append(a)

def grid_max(x,N):
    tmp = -2147000000
    cumsum_diag1=0
    cumsum_diag2=0
    for i in range(N):
        cumsum_row=0
        cumsum_col=0
        for j in range(N):
            cumsum_row+=x[i][j]
            cumsum_col+=x[j][i]

        if cumsum_row > tmp:
            tmp = cumsum_row
            cumsum_row=0
        elif cumsum_col > tmp:
            tmp= cumsum_col
            cumsum_col=0

        cumsum_diag1+=x[i][i]
        cumsum_diag2+=x[i][-(i+1)]

    if cumsum_diag1 > tmp:
        tmp = cumsum_diag1
    elif cumsum_diag2 > tmp:
        tmp = cumsum_diag2
        
    return tmp

print(grid_max(x,n))
