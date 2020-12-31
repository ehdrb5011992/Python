import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
x = list(map(int,input().split()))
def score(x):
    scores = []
    cum_score = 0
    for i in x:
        if i==0: # 틀린경우
            cum_score = 0
            scores.append(0)
        else: # 맞은경우
            cum_score += 1
            scores.append(cum_score)
    return print(sum(scores))

score(x)
