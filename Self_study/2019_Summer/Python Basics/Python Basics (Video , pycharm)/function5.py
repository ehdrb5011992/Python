a=1
b=2
c=-8

def root(a,b,c):
    r1 = (-b + (b**2 -4*a*c)**0.5)/(2*a)
    r2 = (-b - (b**2 -4*a*c)**0.5)/(2*a)

    return r1,r2 #튜플로 저장

r1, r2=root(a,b,c)
print('근은 {} {}'.format(r1,r2))


print("안녕하세요, 저는 \"이동규\" 입니다.")
print("There\'s a cat.")