import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n,m = list(map(int,input().split()))

cnt=[0]*(n+m+3) # 범위 초기화. +3까지 넉넉하게 잡음. n+m일꺼임.
max = -214700000 # 가장 작은값으로 초기화

for i in range(1,n+1):
    for j in range(1,m+1):
        cnt[i+j] += 1

for i in range(n+m+1): # 값이 n+m까지 돌아야함.
    if cnt[i] > max:
        max = cnt[i] # 이렇게해서 최대값을 고정시켜버림.
        
for i in range(n+m+1): # 그리고나서 최대값과 같은거있으면 계속출력.
    if cnt[i] == max:
        print(i, end=' ')
