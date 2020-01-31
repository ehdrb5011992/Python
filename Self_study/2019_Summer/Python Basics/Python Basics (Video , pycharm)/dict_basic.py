wintable = {'가위':'보', #나 : 너
            '바위':'가위',
            '보':'가위'
            }
print(wintable['가위']) #dictionary는 숫자대신 문자

words = ['a','b','c']
print(words[0])
print(words[1])

def rsp(mine,yours) :
    if mine== yours :
        return 'draw'
    elif wintable[mine] == yours :
        return 'win'
    else :
        return 'lose'
result=rsp('가위','바위')
print(result)

messages = {
    'win' : '이겼다!' ,
    'draw' : '비겼네.',
    'lose' : '졌어...'
}
print(messages[result])