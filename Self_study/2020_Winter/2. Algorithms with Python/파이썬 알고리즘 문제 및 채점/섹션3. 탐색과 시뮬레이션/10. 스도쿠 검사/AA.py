import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

x=[list(map(int,input().split())) for _ in range(9)]

def is_zero_nine(x):
    answer=set(range(1,10))
    x=set(x)
    if x == answer:
        return True
    else:
        return False

def transpose(x):
    ans = [[0]*len(x) for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x)):
            ans[i][j] = x[j][i]
    return ans

def line_check(x):
    for row in x:
        if is_zero_nine(row):
            continue
        else:
            return False
    x=transpose(x)
    for col in x:
        if is_zero_nine(col):
            continue
        else:
            return False
    return True

def square_check(x):
    for case_row in range(3):
        for case_col in range(3):
            ans=[]
            for row in range(3*case_row,3*(case_row+1)):
                for col in range(3*case_col,3*(case_col+1)): 
                    ans.append(x[row][col])
            if is_zero_nine(ans):
                continue
            else:
                return False
    return True

def total_check(x):
    if line_check(x) & square_check(x):
        return 'YES'
    else:
        return 'NO'


print(total_check(x))
