########################################################################

#NUMPY


import numpy as np

#a,b는 행렬 혹은 벡터임 //
#np.random.seed(0)
#np.array([2,3,4])
#np.arange(1,10)
#np.linspace(1,10,10)
# * @ / + - a.dot(b)
#np.linalg.inv(a)
#np.linalg.det(a)
#np.linalg.solve(A,b) #Ax=b , x = ?
#a.min(axis = 1) a.max(0) a.sum(1) a.cumsum(1)
#a.reshape(3,4) #a.resize(3,4)
#a.argmax(aixs=0)
#a > 4  -> T or F
#np.floor(3.5)
#a.ravel(1) #열먼저 읽고, a.ravel(0) # 행먼저 읽음.
#a.T
#a.shape = 3,4
#np.vstack np.hstack((a,b)) #a,b가 행렬일때
#np.column_stack np.row_stack((a,b)) #a,b가 벡터일때
# 위 둘은 반드시 묶어서 -> 괄호2개
#np.random.random(2,2)
#a[:,np.newaxis] #벡터를 행렬로 만들어서 취급할때
#a.copy()
#u = np.eye(3)
#np.trace(u)
#np.nan

########################################################################


#PANDAS


import numpy as np
import pandas as pd

#aa = pd.Series([range(4)]) <- 열벡터 , columns옵션은 없음. 1로 고정이기때문.
#dates = pd.date_range('20130101', periods=6)
#df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
# 위의 index는 정수,문자 다 받을 수 있음.
#pd.Categorical(["test","train","test","train"])
#pd.Series(1,index=list(range(4)),dtype='float32')
#df.head() df.tail(3)
#df.index df.columns df.values
#df.describe()
#df.T
#df.sort_index(axis=1, ascending=False) #여기서 index는 행,열 상관없이.
#df.sort_values(by='B') #아래로 내려갈수록 오름차순.
#df['A']
#df[['A','B']]  == df.loc[:,['A','B']]
#df[0:1]
#df.iloc[3] == df.loc[dates[3]] #4번째 행.
#df.iloc[[1,2,4],[0,2]] loc와 iloc는 리스트안에 리스트 2개로 표현. df를 바로들어가면 각각의 리스트들로 그냥 표현.
#df.A #df 데이터의 A컬럼. -> df[df.A > 0]
# s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
# df['F'] = s1 #F변수에 자동생성후 데이터 추가.
#df.F는 불가.
#df.loc[:,'D'] = 5 #바꾸는건 이렇게
#df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
#df1.loc[dates[0:2],'E'] = 1
#df1.dropna(how='any')
#df1.fillna(value=5)
#pd.isna(df1) #이거는 pd임.
#df.mean(0)
#s = pd.Series([1,3,5,np.nan,6,8], index=dates)
#s.shift(2)
#df.apply(np.cumsum, axis=1)
#df.apply(lambda x: x.max() - x.min())
# s = df.iloc[3] ; df.append(s, ignore_index=True) #행들의 인덱스무시

#left = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
#right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5] , 'abs': ['foo', 'bar']} )
#pd.merge(left, right, on='key') #데이터합칠때

# df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
#                    'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
#                    'C' : np.random.randn(8),
#                    'D' : np.random.randn(8)})
# df.groupby('A').sum() #더할수 있는 숫자만
# df.groupby(['A','B']).sum()


#import os
#os.getcwd()
#os.chdir("/Users/82104/Desktop")
#df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
#df.to_csv('foo.csv')
#pd.read_csv('foo.csv', index_col=0)

########################################################################
#MATPLOTLIB

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib import rcParams
# rcParams['figure.figsize'] = [10,6] #[x축,y축] 이게 내노트북이랑 잘맞음.

#t = np.arange(0.0, 2.0, 0.01) #x축 ; s = 1 + np.sin(2 * np.pi * t) #y축
#fig, ax = plt.subplots()
#ax.plot(t, s)
#ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#       title='About as simple as it gets, folks') # 전체를 다 포현. 아래와비교 잘하기
#ax.grid(alpha=0.2) #격자의 두께를 옵션.
#fig.savefig("test.png")

#plt.subplot(2, 1, 1)
#plt.plot(x1, y1, 'ro-')
#plt.title('A tale of 2 subplots') #그냥 각각으로 처리해서 넣을수도있음.
#plt.ylabel('Damped oscillation')

# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('time (s)')
# plt.ylabel('Undamped')
# 히스토그램
# np.random.randn(437) 정규난수생성
# x는 데이터, num_bins는 상수일때,
# n, bins, patches = ax.hist(x, num_bins, density=3)
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#    np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) #그냥 정규곡선임.
# ax.plot(bins, y, '--') #좌표, y값, --선분
# ax.set_xlabel('Smarts') #ax로 했으면 set을 쓰는데 , 그중에서도 xlabel이렇게씀.
#                         #plt.xlabel과 비교해서 알아놓기.
# ax.set_ylabel('Probability density')
# ax.set_title('Histogram of IQ: $\mu=100$, $\sigma=15$')

#########################그 외에 필요하면 나중에 참고.##########################

#SEABORN


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# print("Seaborn version : ", sns.__version__) #현재 버전 체크.
# sns.set() #초기화.
# sns.set_style('whitegrid') #도표의 style을 whitegrid로 줌. 그밖에 darkgrid 등이 있음.

#1 relplot
# tips = sns.load_dataset("tips")
# sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips)
# sns.relplot(x='total_bill',y='tip',size='size',sizes=(15,200),data=tips )
# df = pd.DataFrame(dict(time=np.arange(500), value=np.random.randn(500).cumsum()))
# dict의 문법을 위에처럼 숙지해놓기.
# g.fig.autofmt_xdate() #크기맞춤// 이런게있다 정도로 참고.

#2 catplot
#sns.catplot(x="day", y="total_bill", hue="smoker",col="time", aspect=.6, kind="swarm", data=tips)
#sns.catplot(x='day',y='total_bill',jitter = False,data=tips);
#sns.catplot(x='smoker',y='total_bill',order= ['No','Yes'],data=tips);
#kind = 'box' 'bar' 'line' 'swarm' ...
#len(tips.query('day == "Fri"'))
#g = sns.catplot(x="fare", y="survived", row="class", #가로로보고싶을땐 row옵션.
#                kind="box", orient="h", height=1.5, aspect=4,
#                data=titanic.query("fare > 0 ")) #query를 통해 행속성의 필터링

#3. lmplot&regplot
#sns.regplot(x='total_bill',y='tip',data=tips);
#sns.lmplot(x='total_bill',y='tip',data=tips);
#sns.lmplot(x='size',y='tip',data=tips,x_estimator=np.mean);
#sns.lmplot(x='total_bill',y='tip',hue='smoker',data=tips)
#g = sns.lmplot(x="total_bill", y="tip", row="sex", col="time",
#               data=tips, height=3)
#g = (g.set_axis_labels("Total bill (US Dollars)", "Tip")
#     .set(xlim=(0, 60), ylim=(0, 12),
#          xticks=[10, 30, 50], yticks=[2, 6, 10])
#     .fig.subplots_adjust(wspace=.02))
#g.set(xlim=(0,100),ylim=(0,20),xticks=[10,30,50],yticks=[2,6,10],title="hi",  xlabel = "hello" , ylabel = "hehe")
#sns.regplot(x=df["sepal_length"], y=df["petal_length"], color='red')
#sns.regplot(x=df["sepal_length"], y=df["petal_length"], line_kws={'color': 'red'})
#sns.regplot(x=df["sepal_length"], y=df["petal_length"], scatter_kws={'color': 'red'})
#sns.regplot(x=df["sepal_length"], y=df["petal_length"], fit_reg=False)
# ax = sns.regplot(x=df["sepal_length"], y=df["petal_length"])
# ax.set_ylim([0, 8]) #한계를 정함. 이때 그래프가 창에 띄워져있어야 실행됨.
# ax.set_xlabel('Sepal Length') #그림이 켜져있는 상태에서 실행
# ax.set_xlabel('꽃잎 길이') #한글을 사용하면 글자가 꺠짐.
# plt.rc('font', family='Malgun Gothic') #말건고딬
# plt.rc('axes', unicode_minus=False) #마이너스가 깨지면 이걸 입력하면됨.

#4. pairplot

# iris = sns.load_dataset("iris")
# sns.pairplot(iris)
# g = sns.PairGrid(iris)#밑바탕 출력후,
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot, n_levels=6)

#5. distplot(histogram)

#x=np.random.normal(size=100)
#sns.distplot(x);
#sns.distplot(x,kde=False,rug=True);

#6. jointplot
#sns.jointplot(x='sepal_length',y='sepal_width',data=iris)