import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
x=[]
for i in range(n):
    x.append(input())
    
    
def is_circular(x):
    x = x.lower()
    tmp = x[-1::-1]
    if tmp == x:
        return 'YES'
    else:
        return 'NO'

for idx,word in enumerate(x,1):
    ans=is_circular(word)
    print("#{0} {1:3s}".format(idx,ans))
