import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L):
    global cnt
    if L==n+1:
        for j in res:
            print(chr(int(j)+64), end='')
        print()
        cnt+=1
    else:
        if x[L-1]==0:
            res.append(x[L-2:L-1])
            DFS(L+1)
            
        else:
            res.append(x[L-1])
            DFS(L+1)
            res.pop()
            if (L+1 <= n) and (x[L-1:L+1] <= str(26)):
                res.append(x[L-1:L+1])
                DFS(L+2)
                res.pop()

if __name__=='__main__':
    x=input()
    n=len(x)
    res=[]
    cnt=0
    DFS(1)
    print(cnt)
