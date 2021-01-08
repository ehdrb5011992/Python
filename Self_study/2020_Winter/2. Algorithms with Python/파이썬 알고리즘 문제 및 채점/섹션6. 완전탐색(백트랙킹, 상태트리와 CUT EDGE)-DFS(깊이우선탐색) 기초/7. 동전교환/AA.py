import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


def DFS(L,sum):
    global res
    if L>res:
        return
    if sum > m: # 값이 만약 가지고 제한보다 크면 / 가지뻗으면안됨.
        return
    if sum==m:
        if L < res: # 2. 이미 참조하고 있었음.
            res=L # 1. 지역변수 선언했는데,
    else:
        for i in range(n): # 동전의 개수만큼 돎.
            DFS(L+1,sum+a[i])
                                

if __name__=='__main__':
    

    n=int(input())
    a=list(map(int,input().split()))
    m=int(input())
    res=2147000000
    a.sort(reverse=True) # 가지를 뻗을때는
                         # 기존 데이터를 내림차순으로 정렬 시, 빨리 볼 수 있다.
    DFS(0,0)
    print(res)
