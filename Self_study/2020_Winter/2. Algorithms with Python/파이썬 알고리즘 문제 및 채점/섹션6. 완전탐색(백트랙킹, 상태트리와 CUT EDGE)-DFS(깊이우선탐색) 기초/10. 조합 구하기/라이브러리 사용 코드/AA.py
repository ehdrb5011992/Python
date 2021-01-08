import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

# 다만, 라이브러리를 사용해서 순열,조합을 사용하는데 익숙해지지 말기.
# 조건이 주어지면, 라이브러리 안통함.

import itertools as it # itertools라는 라이브러리를 쓸거다.
n,k=map(int,input().split())
a=list(map(int,input().split()))
m=int(input())
cnt=0
for x in it.combinations(a,k): # a라는 리스트에서 k개를 뽑아서 만드는게 x에대응
    if sum(x)%m ==0:
        cnt+=1
print(cnt)
