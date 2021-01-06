import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


x=input()
stack=[]
res=0

for i in x:
    if i.isdecimal():
        stack.append(i)
    else:
        if stack and (i in '+-*/'):
            n1=stack.pop()
            n2=stack.pop()
            res=eval(n2+i+n1)
            stack.append(str(res))
else:
    print(res)
    
    
