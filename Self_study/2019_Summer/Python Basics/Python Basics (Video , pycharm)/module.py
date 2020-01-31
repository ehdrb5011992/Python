import math #수학 모듈
r=10
2*math.pi*r

math.ceil(2.5)
math.floor(2.5)

import random
candidates = ['가위','바위','보']
selected = random.choice(candidates) #무작위로나옴 (r의 sample함수)
print(selected)

def get_web(url) :
    """ URL을 넣으면 페이지 내용을 돌려주는 함수"""
    import urllib.request
    response=urllib.request.urlopen(url)
    data=response.read()
    decoded=data.decode('utf-8')
    return decoded

url = input('웹 페이지 주소?>')
content = get_web(url)
print(content)

