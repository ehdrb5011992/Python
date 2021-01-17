import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


n=int(input())
arr=[]
while True:
    tmp=list(map(int,input().split()))
    if tmp[0] == -1:
        break
    else:
        arr.append(tmp)

graph= [[1000]*(n+1) for _ in range(n+1)]
for i in range(n+1):
    graph[i][i]=0
for i in range(len(arr)):
    graph[arr[i][0]][arr[i][1]]=1
    graph[arr[i][1]][arr[i][0]]=1

for k in range(n+1):
    for i in range(n+1):
        for j in range(n+1):
            graph[i][j]=min(graph[i][j],graph[i][k]+graph[k][j])

res=[]
for i in range(1,n+1):
    res.append(max(graph[i][1:n+1]))
mini=min(res)
ans=[]
for ind,val in enumerate(res,1):
    if val==mini:
        ans.append(ind)
        
print(mini,len(ans))
for i in ans:
    print(i, end=' ')
