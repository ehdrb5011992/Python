a=1
b=2
c=-8

def print_sqrt(a,b,c):
    r1=(-b + (b**2 -4*a*c)**0.5)/(2*a)
    r2 = (-b - (b**2 -4*a*c)**0.5)/(2*a)

    print('해는 {}또는 {}'.format(r1,r2))

print_sqrt(a,b,c)

def print_round(number) :
    rounded = round(number)
    print(rounded)

print_round(4.6)
print_round(2.2)