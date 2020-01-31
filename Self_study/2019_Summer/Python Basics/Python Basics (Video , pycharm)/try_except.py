list = []
list[0]
text='abc'
number = int(text)

text = '100%'
try :
    number=int(text)
except ValueError : #error의 경우, 별도의 처리를 함.
    print('{}는 숫자가 아니네요.'.format(text))


def safe_pop_print(list,index) :
    try:
        print(list.pop(index))
    except IndexError:
        print('{} index의 값을 가져올 수 없습니다.'.format(index))

safe_pop_print([1,2,3],5)

#if else로도 처리가능
def safe_pop_print(list,index) :
    if index<len(list):
        print(list.pop(index))
    else:
        print('{} index의 값을 가져올 수 없습니다.'.format(index))

safe_pop_print([1,2,3],5)

try:
    import my_module
except ImportError:
    print("모듈이 없습니다.")
#이 경우는 try-except가 아니면 에러를 잡아내기 어렵다.
#try는 조건을 걸지 않기 때문.

a = 3/0
try :
    3/0
except Exception:
    print("0으로 나눌 수 없습니다.")

