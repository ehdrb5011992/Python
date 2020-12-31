import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = int(input())
x = []
for _ in range(n):
    x.append(list(map(int,input().split())))

def price(x):
    spa = [0]*7 # 주사위 (0과 1~6)
    for i in x:
        spa[i]+=1
        
    tmp = -2147000000
    for idx,value in enumerate(spa):
        if value == 3:
            return 10000+idx*1000
        elif value == 2:
            return 1000+idx*100
        elif value == 1:
            pri = idx*100
            if pri>tmp:
                tmp=pri
    return tmp

tmp = -2147000000
for ins in x:
    ins_price=price(ins)
    if ins_price > tmp:
        tmp = ins_price
print(tmp)
