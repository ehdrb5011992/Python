import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text
 
if __name__=="__main__":
    n, m=map(int, input().split()) # n은 노드수, m은 간선의 수
    dis=[[5000]*(n+1) for _ in range(n+1)] # 5000의 큰값으로 초기화함. 최소값 찾는 문제이기 때문.
    for i in range(1, n+1): # 자기자신은 0으로 초기화
        dis[i][i]=0
    for i in range(m):
        a, b, c=map(int, input().split()) # 인접행렬 만드는 for문
        dis[a][b]=c # i,j에서 바로 가는 초기값
    for k in range(1, n+1): # k를 거쳐서 가는거임. k는 행의 순서가 되는거임. (노드의 개수)
        for i in range(1, n+1):
            for j in range(1, n+1):
                dis[i][j]=min(dis[i][j], dis[i][k]+dis[k][j]) # 이렇게 삼중for문 돌리는게 굉장히 중요하다. 잊지말
    for i in range(1, n+1):
        for j in range(1, n+1):
            if dis[i][j]==5000:
                print("M", end=' ')
            else:
                print(dis[i][j], end=' ')
        print()
