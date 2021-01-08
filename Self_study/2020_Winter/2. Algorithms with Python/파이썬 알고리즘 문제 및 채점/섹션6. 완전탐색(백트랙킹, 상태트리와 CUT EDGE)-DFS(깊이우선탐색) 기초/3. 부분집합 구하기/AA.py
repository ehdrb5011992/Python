import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(v):
    if v==n+1: # 3의 부분집합을 구하는경우, 리프 노드로 4를가지면 출력하는거임.
        for i in range(1,n+1): # 공집합인경우는 출력안해도 됨.
            if ch[i]==1: 
                print(i,end=' ') # 그래서 들어와있는 쌍에 따라 출력
        print() # 줄바꿈

    else: # 여기가 재귀문.
        
        ch[v]=1 # 켯음.
        DFS(v+1)
        ch[v]=0 # 껏음.
        DFS(v+1)


if __name__=='__main__':
    n=int(input()) # 마찬가지로, 전역변수로써 사용됨.
    ch=[0]*(n+1) # check변수. 왼쪽 및 오른쪽임. 넉넉히 n+1 // 전역변수로써 외부에씀
    DFS(1)

