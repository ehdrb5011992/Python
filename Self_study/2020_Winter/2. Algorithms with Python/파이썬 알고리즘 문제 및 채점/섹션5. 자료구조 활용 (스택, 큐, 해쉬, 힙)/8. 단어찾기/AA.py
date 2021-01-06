import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n=int(input())
words=[]
poet=[]
for _ in range(n):
    words.append(input())
for _ in range(n-1):
    poet.append(input())

def solution(words,poet):
    word_dict={}
    for i in words:
        word_dict[i]=0
    for i in poet:
        word_dict[i]+=1
    
    for key,value in word_dict.items():
        if value==0:
            return key

print(solution(words,poet))
