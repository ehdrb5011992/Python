import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text
T=int(input()) # case 개수를 읽은거임. 첫째줄을 읽음. 그리고 input을 수행하면 자동 아래로 커서가 옮겨짐. 
for t in range(T):
    n, s, e, k = map(int, input().split()) # 나중에는 변수명 의미있게 작성
    a = list(map(int,input().split())) # 이게 input으로 들어옴
    a=a[s-1:e] # 번째를 조심히 인식하자. 
    a.sort()
    print('#{0:d} {1:d}'.format(t+1,a[k-1]))
