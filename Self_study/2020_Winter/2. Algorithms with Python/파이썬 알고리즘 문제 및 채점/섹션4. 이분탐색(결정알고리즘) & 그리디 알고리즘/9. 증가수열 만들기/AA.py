import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
x=list(map(int,input().split()))

ans=''
num=0
cnt=0
while n:
    s=x[0]
    e=x[-1]
    
    if (len(x)==1) & (x[0]>num) :
        ans+='L'
        cnt+=1
    if num < s and num < e:
        if s<e:
            num=x.pop(0)
            ans+='L'
            cnt+=1
        elif s>e:
            num=x.pop()
            ans+='R'
            cnt+=1
    
    elif num < s and num > e:
        num=x.pop(0)
        ans+='L'
        cnt+=1
        
    elif num > s and num < e:
        num=x.pop()
        ans+='R'
        cnt+=1
    
    else:
        break
    n-=1

print(cnt)
print(ans)
