import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

n = input()
x = list(map(int,input().split()))

def repre_value(x):
    mean_x = round(sum(x)/len(x))
    
    compare_number = x[0]
    index_reposit = 0
    
    for index in range(len(x)):
        if abs(x[index]-mean_x) < abs(compare_number-mean_x):
            compare_number=x[index]
            index_reposit=index
        elif abs(x[index]-mean_x) == abs(compare_number-mean_x):
            if x[index] > compare_number:
                compare_number=x[index]
                index_reposit=index
    return print(mean_x,index_reposit+1)

repre_value(x)




