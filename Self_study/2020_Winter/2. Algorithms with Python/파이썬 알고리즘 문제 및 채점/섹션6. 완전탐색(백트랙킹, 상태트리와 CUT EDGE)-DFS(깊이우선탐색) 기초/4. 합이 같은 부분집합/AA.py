import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

def DFS(L,sum): # L은 index번호, sum은 부분집합의 누적합 (l은 level의 의미가있음)
    
    if sum>total//2: # 여기가 추가됨. 만약 같다라고 놓으면, 홀수만있는경우 문제가 생김.
        return


    if L==n:
        if sum==(total-sum): # 부분집합의 합이 같으면,
            print('YES')
            sys.exit(0) #함수가 아니라, 프로그램이 아예 종료
    else:
        DFS(L+1,sum+a[L]) # 하거나,
        DFS(L+1,sum) # 하지 않거나.
        
if __name__=='__main__':
    n=int(input()) # 마찬가지로, 전역변수로써 사용됨.
    a=list(map(int,input().split()))
    total=sum(a)
    DFS(0,0)
    print('NO')
