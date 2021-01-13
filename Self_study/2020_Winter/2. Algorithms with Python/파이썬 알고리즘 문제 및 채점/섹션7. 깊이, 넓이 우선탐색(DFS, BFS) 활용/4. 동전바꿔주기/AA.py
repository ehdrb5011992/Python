import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,sum):
    global cnt
    
    if sum>t:
        return
    if L==k:
        if sum==t:
            cnt+=1
    else:
        for i in range(n[L]+1):
            DFS(L+1,sum+i*p[L])
        

if __name__=='__main__':
    t=int(input())
    k=int(input())
    p=[]
    n=[]
    for i in range(k):
        tmp1,tmp2=map(int,input().split())
        p.append(tmp1)
        n.append(tmp2)
    cnt=0
    DFS(0,0)
    print(cnt)

