list1= ['가위','바위','보']
list2= [37,23,10,33,29,40]

print(list2)
#list2.append(16) 문자열에 format처럼 쓰는것
print(list2)
list3 = list2 + [16]
print(list3)
list4=list2+list3
print(list4)

n=12
ownership =n in list3
print(ownership)
n=10
if n in list3 :
    print('{}은 있어!'.format(n))

del(list4[12])
list4.remove(40)
print(list4)

list1=[1,2,3]
list2=[4,5,6]
#list3 =list1+list2
list3 = list1.extend(list2)
list1
x=list1.remove(5)
list1

list1+list2
print('a','a')