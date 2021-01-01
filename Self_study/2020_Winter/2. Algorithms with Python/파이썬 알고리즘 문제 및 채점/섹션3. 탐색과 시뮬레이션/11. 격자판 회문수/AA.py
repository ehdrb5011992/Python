import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

x=[list(map(int,input().split())) for _ in range(7)]

def is_circul(x): # x- > 5원소 리스트
    n=len(x)//2    
    if all(x[i]==x[-(i+1)] for i in range(n)): #회문이면 True, 아니면 False
        return True
    else:
        return False

def transpose(x): # x-> 매트릭스
    temp=[[0]*len(x) for _ in range(len(x[0]))]
    for i in range(len(x)):
        for j in range(len(x[0])):
             temp[j][i] = x[i][j]
    return temp

def count_circul(x): # x-> 매트릭스
    
    cnt=0 
    for i in x: # 행기준
        for j in range(3):
            check=i[j:j+5]
            if is_circul(check):
                cnt+=1
    x = transpose(x) 
    for i in x: # 열기준
        for j in range(3):
            check=i[j:j+5]
            if is_circul(check):
                cnt+=1
        
    return cnt

print(count_circul(x))
