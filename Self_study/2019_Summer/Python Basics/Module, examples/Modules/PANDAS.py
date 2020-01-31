#pandas
import numpy as np
import pandas as pd

#object creation
s = pd.Series([1,3,5,np.nan,6,8]) #NaN 출력 봐놓기
s

dates = pd.date_range('20130101', periods=6)
print(dates)
print("***")
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD')) #index는 타입상관없음.
print(df)
#dates = ['1','2','3','4','5','6'] #문자
#dates = [int(i) for i in aa ] #정수 다 인덱스로 받음.


df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'), #type을 실수형 32비트
                     'D' : np.array([3] * 4,dtype='int32'), #정수형 32비트는 초기값.
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
print(df2) #카테고리 빼고 리스트를 써도되나, 그러면 범주가 아님!
print("***")
print(df2.dtypes)

#viewing data
print(df.head())
print("***")
print(df.tail(3))

print(df.index) #인덱스 보여줌.
print("***")
print(df.columns) #컬럼 보여줌
print("***")
print(df.values) #값 보여줌

print(df.describe()) #R의 summary함수.
print(df.T)
print(df.sort_index(axis=1, ascending=False)) #1이면 가로(행), 0이면 세로(열), 왼쪽은 내림차순정렬
print(df.sort_values(by='B')) #아래로 내려갈수록 오름차순.


#selection
print(df['A']) #문자로 쓰면 열들을 뽑으라는것. (기본)
print(df[['A','B']]) # = print(df.loc[:,['A','B']])
print("***")

print(df[0:1]) #행을 뽑고 싶으면 반드시 이렇게 범위로 써야함.
print(df.loc[dates[0]])  #print(df[0:1][:].T) 와 유사. df[행][열]
#데이터로서 표시할때는 이렇게.
print(df.loc[:,['A','B']]) #뽑을때는 loc함수 쓸것.얘는 '이름'으로뽑음.

print(df.loc['20130102':'20130104',['A','B']]) #==print(df[1:4][['A','B']])
#print(df[1:4][1:2])는 [1:4]뽑은것 중 [1:2]를 의미.
#print(df[1:4][['A','B']]) == print(df[1:4][df.columns[0:2]]) 임. 불편하지만 받아들이기.
#==print(df.loc[dates[1:4],['A','B']])
# print(df.loc[1:3,['A','B']]) 얘는 안됨.

print(df.iloc[3]) #== print(df.loc[dates[3]])
print(df.iloc[[1,2,4],[0,2]]) #얘는 'index'로 뽑는거임.
print(df[df.A > 0]) ## df.A는 df변수의 A컬럼이라는 뜻임. 자동지정되므로 참고하기!

print(df)
print("***")
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
df['F'] = s1 #기존 df의 변수에 대해서만 추가를 시킴.
#df.F = s1 생성하면서 동시에 넣는건 위에서나 가능. 왼쪽에서처럼은 불가능.
print(df)

df.loc[:,'D'] = 5 #바꾸는건 이렇게
print(df)



#Missing data
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E']) #index만 바꿔서 다시 데이터를 만들때
df1.loc[dates[0:2],'E'] = 1
print(df1)

print(df1.dropna(how='any')) #NaN하나라도 있으면 버리기
print(df1.fillna(value=5)) #NaN을 값5로 넣기 (데이터변수.na처리함수)
print(pd.isna(df1)) #na가 있는것을 보기. 이때 pd.isna method임을 알고있기.


#Operations
print(df.mean()) # == print(df.mean(0))
print("***")
print(df.mean(1))

s = pd.Series([1,3,5,np.nan,6,8], index=dates)
print(s)
print("***")
print(s.shift(2)) #데이터들을 아래로 2칸 옮김
print("***")
print(s.shift(-1)) #데이터들을 위로 1칸 옮김. 빈칸은 NaN

print(df)
print("***")
print(df.apply(np.cumsum)) # == print(df.apply(np.cumsum, axis=0)) a
print("***")
print(df.apply(np.cumsum, axis=1))
print("***")
print(df.apply(lambda x: x.max() - x.min())) #lambda x는 함수명령어.(아마 매틀랩에서 배운듯)
#즉, 괄호에는 함수가 들어간다는 뜻
#print(df.apply(sum, axis=1)) <--- apply함수는 괄호안의 첫번째에 함수를받고, 두번째에 시행하는방향.
#R의 apply문과 비슷.


# Merge
print(df)
print("***")
s = df.iloc[3] #4번째행이 출력됨.
print(df.append(s, ignore_index=True)) #append명령어를 써서 출력
#print(df.append(s, ignore_index=False)) 이건 인덱스를 고스란히 출력하라는 뜻.
#맨 아래행에 데이터가 추가된다.

left = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5] , 'abs': ['foo', 'bar']} )
print(left)
print("***")
print(right)
print("***")
print(pd.merge(left, right, on='key')) #  print(pd.merge(left, right)) 얘는 최대한 다겹쳐서 출력.
# merge는 항상 기준이 있어야함.
#print(pd.merge(left, right, on='key')) 이러면 중복은 key만 취급. (key가 기준)

#Grouping
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
print(df)

print(df.groupby('A').sum()) #더할수 있게되는 숫자들만 더함.
print(df.groupby(['A','B']).sum()) #A,B순서쌍에 대한 그룹들에서 더함

#Getting Data In/Out
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df.to_csv('foo.csv') #df를 csv로 보낸다 => df.to_csv 를 씀.

pd.read_csv('foo.csv') #read_csv는 pd의 함수.
pd.read_csv('foo.csv', index_col=0) #index로 지정한 col은 0번째라는 뜻.

###참고####
import sys , os
#sys.path.append("/Users/82104/Desktop") #얘를 쓰는게 아니다!
os.chdir("/Users/82104/Desktop") #작업환경은 얘로 바꾼다!
#os.getcwd() #현재경로
#os.path.abspath(path) #절대경로
#os.path.dir(path) #경로중 디렉토리명만 얻기
#os.path.basename(path) #경로중 파일명만 얻기
#등등 os는 https://itmining.tistory.com/122 이사이트를 보기.
