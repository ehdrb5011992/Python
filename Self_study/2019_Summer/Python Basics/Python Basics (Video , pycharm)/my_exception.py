value = '가'

import sys
sys.path  #현재 경로확인 - os.getcwd()로도 가능.
sys.path.append("/Users/82104/.PyCharmCE2019.1/config/scratches/group_study") #얘만 추가해주면 됨. \를 /로 바꾸는것 잊지말기.
# 가장 높은 클래스의 에러는 Exception
from UnexpectedRSPValue import UnexpectedRSPValue

try :
    if value not in ['가위','바위','보'] :
        raise UnexpectedRSPValue #내가 정의한 애러.
except UnexpectedRSPValue: #ValueError를 제외하고 아래를 넣어라.
    print("에러가 발생했습니다.")

#간단한 예) 맥락만 이해하기.
# def sign_up():
#     '''회원가입 함수'''
# try:
#     sign_up()
# except BadUserName: #내가 기존에 정의한 에러 클래스
#     print("이름으로 사용 할 수 없는 입력입니다.")
# except PasswordNotMatched: #내가 기존에 정의한 에러 클래스
#     print("입력한 패스워드가 서로 일치하지 않습니다.")