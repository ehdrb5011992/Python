'''
import sys
sys.path  #현재 경로확인
sys.path.append("/Users/82104/.PyCharmCE2019.1/config/scratches/group_study") #얘만 추가해주면 됨. \를 /로 바꾸는것 잊지말기.
'''

import my_module

selected = my_module.random_rsp()
print(selected)
print('가위?',my_module.SCISSOR == selected)

