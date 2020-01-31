#매우 강력한 기능.

areas = []
for i in range(1,11) :
    areas = areas + [i*i] #append임.
print('areas:',areas)

areas2 = [i*i for i in range(1,11)] #이렇게도 사용가능
#이런 방식을 list_comprehension
print('areas2:',areas2)


areas = []
for i in range(1,11) :
    if i%2 == 0 :
        areas = areas + [i*i]
print('areas:',areas)

areas2 = [i*i for i in range(1,11) if i % 2 == 0] #이렇게도 사용가능
print('areas2:',areas2)

[(x,y) for x in range(15) for y in range(15)] #튜플로도 만들수 있음.


#dictionary comprehension
students = ['태연','진우','정현','하늘','성진']
for number,name in enumerate(students) :
    print('{}번의 이름은 {}입니다.'.format(number,name))

students_dict = {"{}번".format(number+1) : name for number,name in enumerate(students) }
students_dict

scores = [85,92,78,90,100]
students
for x,y in zip(students,scores) :
    print(x,y)
score_dic = {student : score for student,score in zip(students,scores)} #쉽게가능
print(score_dic)


#cf. zip함수
keys1 = ("apple", "pear", "peach")
vals1 = (300, 250, 400)
dict(zip(keys1,vals1))
keys2 = ["apple", "pear", "peach"]
vals2 = [300, 250, 400]
dict(zip(keys2,vals2))

product_list = ["풀", "가위", "크래파스"]
price_list = [800, 2500, 5000]
product_dict = { product : price for product,price in zip(product_list,price_list)}
# product_dict = dict(zip(product_list,price_list)) #이게 더 간단히 표현.
print(product_dict)