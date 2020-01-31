########## 같은부분 #########
list = [1,2,3,4,5]
dict = {'one' : 1 , 'two' : 2}
list[0]
dict['one']

del(list[0])
del(dict['one'])

len(list)
len(dict)

2 in list
7 in list

'two' in dict
'two' in dict.keys()
32 in dict.values()

dict.clear()
dict
list.clear()
list
###########다른부분########

list = [1,2,3,4,5]
dict = {'one' : 1 , 'two' : 2}
list[2]
list.pop(0)
list[2]

dict.pop('one')
dict['two'] #언제나 일정(대응되기때문)

big_list=[1,2,3]+[4,5,6]
dict1={1:100,2:200}
dict2={1:3000,3:300}
dict1.update(dict2)

dict1={1:100,2:200}
dict2={1:3000,3:300}
dict2.update(dict1) #dict2에 dict1로 엎어쳐라
dict2