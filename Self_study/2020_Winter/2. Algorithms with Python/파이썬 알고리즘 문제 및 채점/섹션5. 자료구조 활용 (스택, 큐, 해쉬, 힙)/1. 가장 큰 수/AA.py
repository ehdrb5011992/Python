import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

num, m=map(int,input().split())
num=list(map(int,str(num)))

stack=[]
for x in num:
    while stack and  m>0 and stack[-1] < x:
        stack.pop()
        m-=1
    stack.append(x)

if m!=0:
    stack=stack[:-m]
res=''.join(map(str,stack)) # 이렇게 join사용법을 봐도 됨.
print(res)
