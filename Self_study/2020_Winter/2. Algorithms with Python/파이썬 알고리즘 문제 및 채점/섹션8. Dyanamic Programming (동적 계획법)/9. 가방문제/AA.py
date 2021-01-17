import sys
#sys.stdin=open("input.txt",'rt') # 경로 잡고, 파일 엵음. read a file as text


if __name__=="__main__":
    n,m = map(int,input().split())
    dy=[0]*(m+1)
    for i in range(n):
        w,v=map(int,input().split())
        for j in range(w,m+1): # 미리 공간확보가 필요하기 때문에 j는 w부터돔
            dy[j]=max(dy[j],dy[j-w]+v) # dy[j-w]: w만큼 보석을담기위해서 공간만들어놓음.

    print(dy[m])
