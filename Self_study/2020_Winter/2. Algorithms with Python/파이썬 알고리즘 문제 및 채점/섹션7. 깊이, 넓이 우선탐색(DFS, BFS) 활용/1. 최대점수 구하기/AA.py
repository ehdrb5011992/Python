import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text



def DFS(L):
    global ti_sum,res,tmp
    if ti_sum > m:
        return
    
    if L==n:
        if tmp >= res:
            res=tmp
        
    else:
        ch[L]=1
        ti_sum += x[L][1]
        tmp+=x[L][0]
        DFS(L+1)
        tmp-=x[L][0]
        ti_sum -= x[L][1]
        ch[L]=0
        DFS(L+1)
        



if __name__ == '__main__':
    n,m=map(int,input().split())
    x=[]
    for i in range(n):
        x.append(tuple(map(int,input().split())))
    x.sort(key=lambda x:x[0],reverse=True)
    ch=[0]*n
    ti_sum=0
    tmp=0
    res=-2147000000

    DFS(0)
    print(res)
