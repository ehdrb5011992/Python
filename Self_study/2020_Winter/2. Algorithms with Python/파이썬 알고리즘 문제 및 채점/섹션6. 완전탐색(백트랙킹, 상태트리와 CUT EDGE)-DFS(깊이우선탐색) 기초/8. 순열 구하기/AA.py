import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


def DFS(L):
    global cnt
    if L==m: # 여기는 도달했을때 출력 처리를 어떻게 할것인지에 대한 논의
        for j in range(m): 
            print(res[j],end=' ')
        print()
        cnt+=1
    else:
        for i in range(1,n+1): # 여기를 더 건드리므로써, 가지치기를 선택적으로가능.
            if ch[i]==0:
                ch[i]=1 # 작업을 하기전에 qnfma.
                res[L]=i 
                DFS(L+1) #작업함.
                ch[i]=0 #그리고 다시 역순으로 돌려줘야함
                

if __name__=='__main__':

    n,m=map(int,input().split())
    res=[0]*n
    ch=[0]*(n+1)
    cnt=0
    DFS(0)
    print(cnt)
