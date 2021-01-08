import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,sum,tsum): # L은 index번호 , sum은 지금까지 만든 부분집합의 합
                     # tsum은 

    global result

    if sum +(total-tsum) < result: # 이 문장을 추가해주면 됨.
        return

    if sum>c: 
        return
    
    
    if L==n: 
        if sum > result :
            result=sum 
                       
    else:
        DFS(L+1,sum+a[L],tsum+a[L]) #부분집합에 들어가지 않았어도, 더해줌.
        DFS(L+1,sum,tsum+a[L])
    


if __name__=='__main__': 
    c,n=map(int,input().split())
    a=[0]*n 
    result=-2147000000
    for i in range(n):
        a[i]=int(input())
    total = sum(a) #바둑이 총합 추가
    DFS(0,0,0)
    print(result)
