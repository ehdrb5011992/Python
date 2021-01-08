import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

# 다만, 라이브러리를 사용해서 순열,조합을 사용하는데 익숙해지지 말기.
# 조건이 주어지면, 라이브러리 안통함.

import itertools as it # itertools라는 라이브러리를 쓸거다.
n,f=map(int,input().split())
b=[1]*n
for i in range(1,n):
    b[i]=b[i-1]*(n-i)/i # 이항계수

a=list(range(1,n+1))
for tmp in it.permutations(a): # a의 permutation 조합을 튜플형태로 구해준다.
                               # it.permutations(a,3)은 a리스트에서 3개 퍼뮤테이션

    sum=0
    for L,x in enumerate(tmp):
        sum+=(x*b[L]) #원소곱
    if sum==f:
        for x in tmp:
            print(x,end=' ')
        break # 같으면 종료)
