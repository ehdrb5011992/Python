import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

word1=input()
word2=input()

def solution(word1,word2):
    my_dict={}
    sub_dict={}
    for w1 in word1:
        if w1 not in my_dict:
            my_dict[w1]=1
        else:
            my_dict[w1]+=1

    for w2 in word2:
        if w2 in my_dict:
            my_dict[w2]-=1
        else:
            sub_dict[w2]=1

    if all(x==0 for x in my_dict.values()) and all(x==0 for x in sub_dict.values()):
        return 'YES'
    else:
        return 'NO'

print(solution(word1,word2))
