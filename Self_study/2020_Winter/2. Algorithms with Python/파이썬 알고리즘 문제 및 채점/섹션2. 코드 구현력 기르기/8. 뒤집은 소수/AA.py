import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
num_list = list(map(int,input().split()))

def reverse(x):
    return int(str(x)[-1::-1])

def isPrime(x):
    if x == 1:
        return False
    
    for i in range(2,int(x**(0.5)+1)):
        if x % i ==0 :
            return False
    else:
        return True

for num in num_list:
    rev_num=reverse(num)
    if isPrime(rev_num) == True:
        print(rev_num, end=' ')
