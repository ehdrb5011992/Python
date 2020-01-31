def add_10(value):
    '''value에 10을 더한 값을 돌려주는 함수'''
    if value <10 :
        return 10 #즉시종료
    print('return 뒤')
    result = value +10
    return result

n= add_10(5)
print(n)

n=round(1.5)
print(n)
