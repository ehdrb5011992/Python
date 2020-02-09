"""
참고교재: 텐서플로로 배우는 딥러닝 (박혜정, 석경하, 심주용, 황창하)

예제 2-3-1: Mountain Car에 적용한 DQN 예제

예제는 언덕을 자유롭게 움직일 수 있는 검은색 자동차와 자동차가 도착해야 하는 지점을 
깃발로 표시한 Mountain Car 예제이다. 이 예제의 목적은 자동차가 오른쪽 언덕을 올라가 
깃발 지점에 도착하는 것이다. 그러나 자동차는 언덕 위로 올라갈 만큼의 힘을 가지고 있지 않다. 

에이전트는 오른쪽 언덕을 올라가기 위해 자동차가 충분한 추진력을 얻는 방법을 학습하며, 
추진력을 얻기 위해 자동차는 오른쪽과 왼쪽으로 움직이는 과정을 반복해야 한다.

∙ learning_rate: 신경망의 학습률 매개변수
∙ gamma: 보상을 시간에 따라 얼마나 할인할지 결정하는 할인율 매개변수
∙ n_features: 환경으로부터 받은 에이전트의 입력값으로 자동차의 위치(0)와 속도(1)
∙ n_actions: 에이전트가 선택한 행동으로서 왼쪽(0) 또는 오른쪽으로 밀기(2)와 보류(1) 3가지
∙ epsilon: 탐욕 정책 시 활용되는 탐욕의 초깃값
∙ batch_size: 신경망 학습을 위해 재생 메모리로부터 추출되는 표본의 크기
∙ experience_counter: 현재 재생 메모리에 저장된 표본의 수
∙ experience_limit: 재생 메모리의 최대 용량
∙ replace_target_pointer: 목표 신경망(target network) 갱신 기준 학습단계
∙ learning_counter: 기본 신경망(primary network)의 학습단계
∙ memory: 재생 메모리의 초깃값(0으로 설정)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

import gym #pip install gym

class DQN():
    def __init__(self, learning_rate, gamma, n_features, n_actions, epsilon, parameter_changing_pointer, memory_size):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.batch_size = 100
        self.experience_counter = 0
        self.experience_limit = memory_size
        self.replace_target_pointer = parameter_changing_pointer
        self.learning_counter = 0
        self.memory = np.zeros([self.experience_limit,self.n_features*2+2])
        self.hidden_units = 10

        self.model = self.build_network()
        self.target_model = self.build_network()

    def build_network(self):
        model = tf.keras.Sequential()
        model.add(Dense(self.hidden_units, input_shape = (self.n_features,),activation='relu', 
                        kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
        model.add(Dense(self.n_actions, activation='linear', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
        model.compile(loss='mse', optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def store_experience(self,obs,a,r,obs_):
        index = self.experience_counter % self.experience_limit
        self.memory[index,:] = np.hstack((obs,[a,r],obs_))
        self.experience_counter+=1

#---------------------------------------------------------------
    #실제 학습이 진행되는 단계: 재생 메모리에 저장된 과거 경험을 설정한 배치 크기만큼
    #랜덤으로 추출하여 학습 데이터 집합으로 설정한다. 여기서 epsilon이 조정되며 
    #학습이 진행되는데, 0.9 미만인 경우 epsilon은 학습단계마다 0.0002씩 증가한다.
    def fit(self):
        # sample batch memory from all memory
        if self.experience_counter < self.experience_limit:
            indices = np.random.choice(self.experience_counter, size=self.batch_size)
        else:
            indices = np.random.choice(self.experience_limit, size=self.batch_size)

        batch = self.memory[indices,:]

        #qt = target, qeval = main model 여기선 predict value 넣어주는거.
        qt = self.model.predict(batch[:,-self.n_features:])
        qeval = self.model.predict(batch[:,:self.n_features])

        qtarget = qeval.copy()

        batch_indices = np.arange(self.batch_size, dtype=np.int32)
        actions = self.memory[indices,self.n_features].astype(int)
        rewards = self.memory[indices,self.n_features+1]
        qtarget[batch_indices,actions] = rewards + self.gamma * np.max(qt,axis=1)

        self.model.fit(batch[:,:self.n_features], qtarget)

        if self.epsilon < 0.9:
            self.epsilon += 0.0002
            
#---------------------------------------------------------------
        #학습 시 기본 신경망의 가중치를 가져와 목표 신경망의 가중치를 갱신한다. 목표 신경망의 
        #가중치 갱신은 현재 기본 신경망의 학습단계(learning_counter)가 목표 신경망의 
        #갱신 기준 학습단계(replace_target_pointer)로 정확히 나누어지는 단계마다 진행된다. 
        #예를 들어, replace_target_pointer를 500으로 설정한다면, 학습이 500번 진행될 때마다 
        #목표 신경망의 가중치가 갱신된다.
        if self.learning_counter % self.replace_target_pointer == 0:
            self.target_model.set_weights(self.model.get_weights())
            print("target parameters changed")

        self.learning_counter += 1

#---------------------------------------------------------------
    #탐욕 정책을 통해 행동을 선택하는 함수: 훈련 초기에는 무작위로 행동을 선택하지만, 
    #훈련이 진행되고 epsilon 값이 높아짐에 따라 모형의 예측에 따라 행동을 선택한다.
    def epsilon_greedy(self,obs):
        #epsilon greedy implementation to choose action
        if np.random.uniform(low=0,high=1) < self.epsilon:
            return np.random.choice(self.n_actions)    
        else:
            return np.argmax(self.model.predict(obs[np.newaxis,:]))

#---------------------------------------------------------------
#다음은 DQN 객체를 생성하여 에이전트를 학습시키고 결과를 도출하는 단계이다. 
#우선, gym. make 함수를 통해 OpenAI GYM에서 MountainCar-v0 환경을 불러온다. 
#환경을 불러온 후 DQN 객체를 생성하여 에이전트를 학습시킨다. 매개변수 설정은 다음과 같다. 
#즉, 신경망의 학습률을 0.001로, 할인율 감마를 0.9로, 입력변수의 개수를 2로, 
#행동의 개수를 3으로, epsilon의 초깃값을 0으로, 목표 신경망 갱신 기준 학습단계를 500으로, 
#재생 메모리의 최대 용량을 5,000으로 설정한다. 그리고 총 10단계의 에피소드를 설정하여 
#학습을 수행하고 결괏값을 반환받는다.
            
if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    #make model p = main, q = target
    dqn = DQN(learning_rate=0.001, gamma=0.9, n_features=env.observation_space.shape[0], 	 
              n_actions=env.action_space.n, epsilon=0.1, parameter_changing_pointer=500,
                memory_size=5000)

    episodes = 10
    total_steps = 0

    for episode in range(episodes):#similar to epoch
        steps = 0		#current time
        obs = env.reset()
        episode_reward = 0 #score
        while True:
            env.render()
            action = dqn.epsilon_greedy(obs)
            print(action)
            obs_,reward,terminate, _ = env.step(action)
            reward = abs(obs_[0]+0.5)
            dqn.store_experience(obs,action,reward,obs_)

            if total_steps > 1000:
                dqn.fit()

            episode_reward += reward
            if terminate:
                break
            obs = obs_
            total_steps += 1
            steps += 1

        print("Episode {} with Reward : {} at epsilon {} in steps {}".
			format(episode+1,episode_reward,dqn.epsilon,steps))

    while True:  
        env.render()	


