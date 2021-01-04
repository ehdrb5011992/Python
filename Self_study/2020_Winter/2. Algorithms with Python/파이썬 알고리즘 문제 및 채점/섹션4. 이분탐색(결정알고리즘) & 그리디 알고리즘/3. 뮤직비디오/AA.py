import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

N,M = map(int,input().split())
x=list(map(int,input().split()))
line=[]
largest=0
def check_sum(x,mid,M):
    tot=0
    dvd=[]
    for i in x:
        if tot+i > mid: #이렇게 하는거임!!!! 매우중요
            dvd.append(tot)
            tot=i
        else:
            tot+=i
    else: # 그리고 나서, 부족한건 for else로 처리할생각하면됨.
        dvd.append(tot)
        
    if len(dvd) <= M: # 조건을 전부 통과시킴
        return True
    else:
        return False


def music(x,M):
    lt=1
    rt=sum(x)
    ans=0
    while lt<=rt:
        mid=(lt+rt)//2
        if check_sum(x,mid,M):
            ans=mid # 조건을 통과하고 나온 최종 mid값이 ans
            rt = mid-1
        else:
            lt = mid+1
    return ans

print(music(x,M))

