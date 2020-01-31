import datetime
datetime.datetime.now() #년 월 일 시 분 초 마이크로초
start_time  = datetime.datetime.now()
type(start_time)
start_time = start_time.replace(year=2020, month=2 , day=1) #자기자신을 바꿀때
start_time #이렇게 바꿀수 있음.

#처음부터 내가 원하는 시간으로 만들때
start_time = datetime.datetime(2020,2,1,0,0,0,0)
start_time

#데이트타임은 빼기연산을 지원함
how_long = start_time - datetime.datetime.now()
type(how_long)
how_long.days
how_long.seconds #시와 분은 얘로 계산 - 지원을 안해줌.
"2월 1일까지는 {}일 {}시간이 남았습니다." . format(how_long.days,how_long.seconds//3600)

#### time_delta

import datetime
hundred = datetime.timedelta(days = 100)
datetime.datetime.now() + hundred #클래스가 달라도 더할수 있게 만들어놓음
type(datetime.datetime.now())

hundred_before = datetime.timedelta(days = -100)
datetime.datetime.now() + hundred_before
datetime.datetime.now() - hundred

tomorrow = datetime.datetime.now().replace(hour=9,minute=0,second=0) + datetime.timedelta(days=1)
