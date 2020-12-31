import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text

x = int(input())

def eratos(x):
    cur_num = set(range(2,x+1))
    my_prime = set()

    for i in range(2,x+1):
        prime=cur_num.pop()
        my_prime.add(prime)
        cur_num -= set(range(i,x+1,i))
        
        if cur_num == set():
            break
            
    return len(my_prime)
print(eratos(x))
