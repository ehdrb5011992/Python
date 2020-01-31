a=1
b=2
c=-8

def print_sqrt(a,b,c):
    r1=(-b + (b**2 -4*a*c)**0.5)/(2*a)
    r2 = (-b - (b**2 -4*a*c)**0.5)/(2*a)

    return print('근1는 %1.2f , 근2는 %1.0f 이다.' %(r1,r2))

print_sqrt(a,b,c)