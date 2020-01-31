#1 . medium

#join(["oven", "envier", "erase", "serious"]) ➞ ["ovenvieraserious", 2]
#join(["move", "over", "very"]) ➞ ["movery", 3]
#join(["to", "ops", "psy", "syllable"]) ➞ ["topsyllable", 1]

# "to" and "ops" share "o" (1)
# "ops" and "psy" share "ps" (2)
# "psy" and "syllable" share "sy" (2)
# the minimum overlap is 1
#join(["aaa", "bbb", "ccc", "ddd"]) ➞ ["aaabbbcccddd", 0]

x = ["ovenen", "enenvier", "erase", "serious"]
x = ["move", "over", "very"]
x = ["to", "ops", "psy", "syllable"]
x = ["aasfeded", "eded12312", "cswe", "cswe2314"]
def join(x) :
    try:
        if isinstance(x,list):
            nums = [] # 초기값
            words = x[0]
            for i in range(len(x) - 1): #리스트 내의 단계들에 대해서 함. 리스트항목이 4개있으면 3번을 수행해야함.
                count = min(len(x[i]), len(x[i + 1]))  # 초기값. 위에서 내려감
                while True: #계속실행.

                    if x[i][-count:] == x[i + 1][:count]: #만약 뒤의값이 같으면 나옴.
                        words += x[i + 1][count :] #words에 단어를 잇고
                        nums.append(count) #연결된 count를 출력
                        break #그리고 루프나옴.
                    elif count == 0: #만약 count==0이될때까지 안되다가 0이되면
                        words += x[i+1] #그냥이어버리고
                        nums.append(count) #그때의 카운트를 출력.
                        break #그리고 나옴.

                    count -= 1 #하나씩 줄이면서 살펴볼것

            num = min(nums) #그렇게 얻어진 모든 count들 중에서 가장 작은 값.
            return print([words, num]) #그리고 출력
        else:
            raise Exception
    except Exception:
        print('주의! 리스트를 입력해주세요.')
join(x)


#2 . very hard

#sexagenary(1971) ➞ "Metal Pig"
#sexagenary(1927) ➞ "Fire Rabbit"
#sexagenary(1974) ➞ "Wood Tiger"

def sexagenary(year) :
    try:
        if isinstance(year,int) :
            stem = ['Wood', 'Fire', 'Earth', 'Metal', 'Water']
            branch = ['Rat','Ox','Tiger','Rabbit','Dragon','Snake','Horse',
                      'Sheep','Monkey','Rooster','Dog','Pig']
            stem2 = sum([[i]*2 for i in stem],[])  #stem을 각각 2번씩 출력.
            branches = branch *5 #branch 60개짜리

            import operator #요소별 덧셈을 위한 모듈
            items_sub = [i + ' ' for i in stem2] *6 #두번씩 나눠서 해야한다.
            items = list(map(operator.add,items_sub,branches)) #역시 같은이유.

            init = year-1984 #초기치를 0으로 만드는 작업. (1984년의 순서를 0으로만듦.)
            if init >= 60 or init < 0 : #60사이클이므로 60보다 클경우와, 작은경우로 나눈다. (2044년 이후, 1984년 전)
                init = init % 60 #나머지를 다뤄주면 된다.

            zodiac = items[init]

            return zodiac

        else :
            raise Exception
    except Exception:
        print("주의! 연도를 넣어주세요.")


def sexagenary2(year) :
    try:
        if isinstance(year,int) :

            stem = ['Wood', 'Fire', 'Earth', 'Metal', 'Water']
            branch = ['Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake', 'Horse',
                      'Sheep', 'Monkey', 'Rooster', 'Dog', 'Pig']

            stem2 = sum([[i]*2 for i in stem ],[])
            init = year - 1984

            first = init % 10
            end = init % 12

            return print(stem2[first],branch[end])
        else :
            raise Exception
    except Exception:
        print("주의! 연도를 넣어주세요.")


sexagenary(4400)
sexagenary(1984)
sexagenary(32)
#3. expert

# XO("ooxx") ➞ true
# XO("xooxx") ➞ false
# XO("ooxXm") ➞ true
# // Case insensitive.
# XO("zpzpzpp") ➞ true
# // Returns true if no x and o.
# XO("zzoo") ➞ false

def XO(x) :
    try:
        if isinstance(x , str):
            ox=list(x) #그냥 바로 x가 str이면 list(x.lower())로 내리고 시작해도됨. 그러면 아래 if문에서 제대로 걸러짐.
            o_num = 0
            x_num = 0
            for i in ox :
                if i.lower() == "o":
                    o_num += 1
                elif i.lower() == 'x' :
                    x_num += 1

            if o_num == x_num :
                return True
            else:
                return False
        else:
            raise Exception

    except Exception:
        print("문자열을 입력하세요")

