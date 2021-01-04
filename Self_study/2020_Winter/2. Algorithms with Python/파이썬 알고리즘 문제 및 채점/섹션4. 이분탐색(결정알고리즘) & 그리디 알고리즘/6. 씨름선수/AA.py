import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
x=[]
for i in range(n):
    x.append(tuple(map(int,input().split())))
    
sorted_x=sorted(x,key=lambda x: (x[0],x[1]),reverse=True)

cnt=1
wt_init=sorted_x[0][1]
for i in range(1,n):
    wt=sorted_x[i][1]
    if wt > wt_init:
        cnt+=1
        wt_init=wt
print(cnt)
