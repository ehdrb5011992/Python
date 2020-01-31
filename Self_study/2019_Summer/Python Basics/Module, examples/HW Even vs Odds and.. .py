#1. Even vs Odds (very hard)
#available_spots([0, 4, 6, 8], 9) ➞ 0
# 9 likes NONE of the following spots: [0, __, 4], [4, __ , 6], [6, __, 8] b/c all of his neighbors are even.
#available_spots([0, 4, 6, 8], 12) ➞ 3
# 12 likes ALL of the spots.
#available_spots([4, 4, 4, 4, 5], 7) ➞ 1
# 7 dislikes every spot except the last one at: [4, __, 5].
#available_spots([4, 4], 8) ➞ 1

######################## 요약 ############################
# odd likes (odd,even) , (even,odd) , (odd,odd) -> hates (even,even)
# even likes (even,odd) , (odd,even) , (even,even) -> hates(odd,odd)
# ==> 짝수판별 함수중 (T,F,T) , (F,T,F) 만 못들어감.

def available_spots(L,x) :
    try:
        if not isinstance(L,list) or not isinstance(x,int) : #L이 리스트, x가 정수일때만 실행
        #얘는 if not( isinstance(L,list) and isinstance(x,int) ) 와 같음.
            raise Exception

        def even_check(num) : #짝수 체크 함수 - 이로써 동시에 홀수도 판별한다
            if num % 2 == 0 :
                return True # 짝수
            else:
                return False # 홀수

        # def even_check(num) : #위의 짝수판별함수는 이와 같다.
        #     if num % 2  :
        #         return False
        #     else:
        #         return True

        L_Bool = [even_check(i) for i in L] #리스트를 짝수함수 기준 T,F로 바꿈.
        x_Bool = even_check(x) #집어넣을 숫자를 짝수함수 기준 T,F로 바꿈.

        count = 0
        for j in range(len(L)-1) :
            temp = L_Bool[j:min(j+2,len(L))] #2개씩만 출력 // min을 쓴 이유는 2개씩 자를때 최대 한계를 정하기 위함.
            temp.insert(1,x_Bool) #가운데 방에 들어감.

            if not (temp == [True,False,True] or  temp == [False,True,False])  :
            # 이 문법은 if not temp == [True,False,True] and not temp == [False,True,False] 과 같음.
                count += 1 # 하나씩 더함.
            else :
                pass #아니면 그냥 통과

        return print(count)

    except Exception :
        print("available_spots(a,b)에 a는 리스트, b는 정수를 넣어주세요.")

available_spots([0, 4, 6, 8], 9)
available_spots([0, 4, 6, 8], 12)
available_spots([4, 4, 4, 4, 5], 7)
available_spots([4, 4], 8)



#2 Valid Name (expert)


# valid_name("H. Wells") ➞ True
# valid_name("H. G. Wells") ➞ True
# valid_name("Herbert G. Wells") ➞ True
# valid_name("Herbert") ➞ False
# Must be 2 or 3 words
# valid_name("h. Wells") ➞ False
# Incorrect capitalization
# valid_name("H Wells") ➞ False
# Missing dot after initial
# valid_name("H. George Wells") ➞ False
#  annot have: initial first name + word middle name
# valid_name("H. George Wasdf.") ➞ False
# Last name cannot be initial

def valid_name(x) :
    try:
        if not isinstance(x,str) : #str아니면 다 거르고보자.
            raise Exception
        else :
            pass

        #기초작업
        temp = list(x) #띄어쓰기가 안 되어 있으면 띄어쓰기를 부여하자. 이때 "."을 기준으로하고, 마지막"."에 대해서는 띄어쓰기를 부여하지 않는다.
        len_temp = len(temp)
        for i in range(len_temp) :
            if temp[i] == '.' and (i != len_temp - 1) : #마지막 제외.
                if temp[i+1] != ' ' :
                    temp.insert(i+1,' ') #이부분이 띄어쓰기 자동삽입
                else :
                    pass #되어있으면 ok
            else :
                continue #이자리 사실 pass 들어와도 상관없음.
        x = "".join(temp) #띄어쓰기 만들어서 붙였다. /// -> 밑작업 끝. 이로써 .뒤에는 무조건 띄어쓰기가 오고, 각각을 이름으로써 인식.

        #### start ####
        names = x.split(" ") #띄어쓰기 단위로 쪼개고 이름들을 리스트로 만들고 보자.
        len_names = len(names) #이름의 길이를 변수로 설정.

        # 첫번째 이름, 두번째 이름에 .이 있는지 혹은 이름이 긴지 판단하는 코드 +
        # 각 이름의 첫글자가 대문자로 시작하는지 판단하고 결론내리는 코드 +
        # 마지막이름 (성) 에 길이가 긴지, 혹은 . 이 있는지 판단하고 결론짓는 코드

        ################################################################################################################
        #코어 함수 정의.
        def core(y): #첫번째, 중간 이름에 대한 코어함수.
            import string
            if string.capwords(y) == y : #string.capwords는 각 띄어쓰기를 구분자로 갖는 단어들의 첫번째 string을 대문자로 출력하는 함수. d
                                         #맞는것들만 거르고 시작.
                                         #혹은 y.capitalize() == y 로 해도 무방.
                if len(y) >= 2: # 길이가 2인 글자들에 대해서
                    if y[1] == '.' :
                        return True #첫 이름 두번째 위치에 .이 있을때 True
                    elif y[len(y)-1] != '.' :
                        return True  #글자길이가 2인경우는 이경우 True지만 3인경우는 False.
                    else :
                        return False #그 외는 전부 False
                else :
                    return False #글자가 1이하면 이유불문 무조건 False
            else :
                return False #글자들의 맨 첫글자가 대문자가 아니면 False

        def sub_core(z) : #마지막이름(성) 에 대한 코어함수.
            if z.capitalize() == z :
                if len(z) >= 2: # 길이가 2 이상인 글자들에 대해서
                    if "." in z :
                        return False #만약 .이 하나라도 있으면 바로 False임.
                    else :
                        return True #그 외는 전부 True. 위의 core함수와 살짝 다르다.
                else :
                    return False #글자가 1이하면 이유불문 무조건 False
            else :
                return False #글자들의 맨 첫글자가 대문자가 아니면 False

        ################################################################################################################

        #아래의 코드는 위에서 정의한 코어함수를 사용해 간단하게 표현.
        if len_names == 2 :
            if sub_core(names[1]) and "." not in names[0] :
                return False
            elif sub_core(names[1]): #마지막이름(성)이 조건들을 만족하면 다음실행
                return core(names[0]) #위에서 정의한 핵심함수 . 1번째 이름에서 얻어진 결과로 출력.
            else:
                return False #마지막이름(성)이 조건을 만족하지 않으면 False

        elif len_names == 3 :
            check = core(names[0]) and core(names[1])
            if sub_core(names[2]): #마지막이름(성)이 조건들을 만족하면 다음실행
                if (names[0][1]=="." and names[1][1] == '.')  : #둘다 .인 경우는 가능. // 이 아래는 벤다이어그램 잘 그려볼것.
                    return check #1번째 이름과 2번째 이름에서 얻어진 결과의 교집합만 출력.
                elif names[1][1] ==  '.'  : #그 외에, 뒤에만 .인경우도 인정.
                    return check
                elif names[0][1] == '.' : #이경우만 예외적으로 뺴준다. 첫, 중간이름이 연동되어있는 경우이기 때문. ("H. George Wells" )
                    return False
                elif  len(names[0]) >= 2 and len(names[1]) >= 2 and len(names[2]) >=2 : #모두다 글자가 이름으로 길이가 있는 경우에만 해당.  ("Herbert George Wells")
                    return check #그때도 역시 위에서 정의한 함수에 의존하는 결과값.
                else :
                    return False #initial(1) first name + word(2+) middle name 의 경우를 포함한 나머지 False
            else :
                return False #마지막이름(성)이 조건을 만족하지 않으면 False

        else :
            return False #만약 이름이 4개이상 혹은 1개이하면 False

    except Exception :
        print("valid_name(x)에 x는 str타입을 넣어주세요")

#다양한 실험.
#1글자 , 4글자 이상
valid_name("HweGe") #F
valid_name("HweGe sdf sdfsda asdf") #F
valid_name("HweGe. sdf sdfsda asdf") #F
#2글자
valid_name("Hwe Ge") #T
valid_name("Hwe.Ge") #F
valid_name("H.Ge") #T
valid_name("H. Ge") #T
valid_name("h. Wells") #F
valid_name("H. wells") #F
valid_name("h. Wells") #F
valid_name("H Wells") #F
#3글자
valid_name("H. G Wells ") #F
valid_name("H. G. Wells") #T
valid_name("Herbert G. Wells") #T
valid_name("H. George Wells") #F
valid_name("Herbert George Wells") #T
valid_name("He. George Wells") #F
valid_name("H. Gsef. Wells") #F
valid_name("h. gsef Wells") #F
valid_name("H. George Wasdf.") #F
valid_name("H. G. W.") #F
valid_name("Herb. G. W") #F

####혜민이형 답####
def valid_name(str):
    names = str.split(" ")
    if len(names) < 2 or len(names) > 3:
        print("break Rule#0 : 단어 수 오류")
        return False
    for i in range(len(names)):
        if names[i][0].islower():
            print("break Rule#1 : 이니셜은 대문자")
            return False
    for i in range(len(names)-1):
        if names[i][-1] == ".":
            if len(names[i]) != 2:
                print("break Rule#2 : 이니셜은 .으로 끝")
                return False
        else:
            if len(names[i]) < 2:
                print("break Rule#3 : 이니셜이 아닐땐 두자 이상")
                return False
    if len(names) == 3:
        if len(names[0]) == 2 and len(names[1]) !=2 :
            print("break Rule#4 : 성이 이니셜이면 미들도 이니셜")
            return False
    if "." in names[-1] or len(names[-1]) < 2:
        print("break Rule #5 : 이름은 반드시 단어")
        return False
    return True
