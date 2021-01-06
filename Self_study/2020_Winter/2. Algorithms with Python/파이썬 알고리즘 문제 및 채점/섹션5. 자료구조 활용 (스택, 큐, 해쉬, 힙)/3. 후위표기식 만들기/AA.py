import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


a=input()
stack=[]
res=''

for x in a:
    if x.isdecimal():
        res+=x
    else: # 직관적으로 이렇게 가자. 숫자인지, 아닌지를 일단 나눠야함.
        if x=='(': # 무조건 (는 stack에 더해야함.
            stack.append(x)
        elif x=='*' or x=='/': # *와 /라면,
            while stack and (stack[-1]=='*' or stack[-1]=='/'):
                res+=stack.pop() # 계속 stack을 뒤에서 하나씩뽑아서 더해줌.
            stack.append(x)
        elif x=='+' or x=='-': 
            while stack and stack[-1]!='(':
                res+=stack.pop()
            stack.append(x)
        elif x==')':
            while stack and stack[-1]!='(':
                res+=stack.pop()
            stack.pop() # (를 뽑아내야함.

while stack: # 나머지 stack에 있는 연산자 뽑아냄
    res+=stack.pop()

print(res)
