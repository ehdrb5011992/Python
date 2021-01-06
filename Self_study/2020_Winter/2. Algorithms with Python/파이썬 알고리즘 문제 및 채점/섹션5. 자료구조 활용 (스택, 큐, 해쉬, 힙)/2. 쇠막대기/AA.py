import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

parenthesis=input()
a=list(parenthesis.replace('()','#'))



def solution(a):
    cnt=0
    res=0
    for i in a:
        if i == '#':
            res+=cnt
        elif i=='(':
            cnt+=1
        else : # i가 ) 인 경우 
            cnt-=1
            res+=1
    return res

print(solution(a))
