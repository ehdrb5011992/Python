import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

# 앞으로 재귀함수 이름은 DFS로 하겠음.
def DFS(x):
    if x==0:
        return  # 함수를 종료시켜라~ 라는 명령도 있음.
    else:
        #print(x%2,end='') # 스택의 개념이다. 잊지말기.
        DFS(x//2)
        print(x%2,end='') # 스택의 개념이다. 잊지말기.
                           # 그래서 재귀함수 아래로 내리면, 순서가 바뀜.
                           # 이런걸 back tracking이라고도 부른다

if __name__=='__main__':
    n=int(input())
    DFS(n)

