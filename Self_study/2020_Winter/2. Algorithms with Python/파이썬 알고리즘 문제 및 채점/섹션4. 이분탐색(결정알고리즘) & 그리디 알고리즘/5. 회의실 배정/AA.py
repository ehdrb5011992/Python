import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
x=[]
for _ in range(n):
    x.append(tuple(map(int,input().split())))

sorted_x=sorted(x, key=lambda x:(x[1],x[0])) # 정렬순서가 x[1]번째 후에 x[0]번째
cnt=1
s,e=sorted_x[0]
for i in range(1,n):
    if sorted_x[i][0] >= e :
        cnt+=1
        s,e=sorted_x[i]
print(cnt)
