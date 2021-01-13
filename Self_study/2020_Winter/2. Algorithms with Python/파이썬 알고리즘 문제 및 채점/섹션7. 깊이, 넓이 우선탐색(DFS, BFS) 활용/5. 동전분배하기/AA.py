import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,sa,sb,sc):
    global res
    
    if L==n:
        if sa==sb or sa==sc:
            return
        elif sb==sc:
            return
        
        lar=max(sa,sb,sc)
        sm=min(sa,sb,sc)
        tmp=lar-sm
        if res>tmp:
            res=tmp
    else:
        DFS(L+1,sa+x[L],sb,sc)
        DFS(L+1,sa,sb+x[L],sc)
        DFS(L+1,sa,sb,sc+x[L])

if __name__=='__main__':
    n=int(input())
    x=[]
    for i in range(n):
        x.append(int(input()))
    res=2147000000
    DFS(0,0,0,0)
    print(res)

