seasons = ['봄','여름','가을','겨울']
for season in seasons :
      print(season)

ages = {'Tod' : 35, "Jane" : 23 , 'Paul' : 62}
print(ages)

for key in ages: #얘도 key만 출력
      print(key)

for key in ages.keys() : #key만 출력
      print(key)
for value in ages.values() : #value만 출력
      print(value)

for key in ages.keys():
      print('{}의 나이는 {}입니다.'.format(key,ages[key]))

for key,value in ages.items():
      print('{}의 나이는 {}입니다.'.format(key,value))
#tip !! 딕셔너리는 출력의 순서를 지키지 않는다. 근데 파이썬 3.6에서는
#딕셔너리도 순서를 지켜준다는것 같은데, 확실하지는 않다.

