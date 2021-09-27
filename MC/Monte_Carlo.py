# 参考https://github.com/nicolas0dupuy/CardPole_w_MC
import gym
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, scale):
        self.action_value = {}  # {[s, a]: [count, value]}
        # store every experience of this episode (the length of which is the 'Return(G)')
        self.episode_buffer = []  # [experience1, experience2, ...]
        self.experience = namedtuple("experience", ["state", "action", "reward"])
        self.scale = scale  # 用于控制状态离散化

    def discrete(self, state):   # 状态空间非离散，可以通过设置几个离散动作点，将其一定范围内的状态值离散化简单处理
        state = (state * self.scale).astype(int)
        return tuple(state)

    def store_experience_in_one_episode(self, state, action, reward):
        state = self.discrete(state)
        exp = self.experience(state, action, reward)
        self.episode_buffer.append(exp)

    def clear_buffer(self):
        self.episode_buffer = []

    def choose_action(self, state, epsilon):
        state = self.discrete(state)
        optimal_score = 0

        for key, value in self.action_value.items():
            if state == key[0]:
                if value[1] > optimal_score:
                    optimal_score = value[1]
                    optimal_action = key[1]
        # 采用e-greedy保证一定的探索性，更好的方法是随实验次数增多探索性降低
        if np.random.random() < epsilon:
            return np.random.randint(2)
        else:
            if optimal_score == 0:  # 还没有该state对应的value信息时，为该位置随机选择动作
                return np.random.randint(2)
            else:
                return optimal_action

    def update_q_value(self):
        lasting_time = len(self.episode_buffer)
        for i in range(lasting_time):
            state, action, _ = self.episode_buffer[i]
            score = lasting_time - i    # 该(s,a)对应的return
            if (state, action) in self.action_value.keys():     # 如果q表有该(s,a)对应的历史值记录
                count, value = self.action_value[(state, action)]
                value = (count * value + score) / (count + 1)
                count += 1
                self.action_value[(state, action)] = [count, value]
            else:
                count = 1
                value = score
                self.action_value[(state, action)] = [count, value]

    def plot(self, episodes, score, colors):
        plt.figure(1)
        x = [i for i in range(episodes)]
        plt.plot(x, score, colors, label='Monte Carlo, epsilon=0.2')     # TODO:可以采取一些曲线优化方法让曲线表现得更加平稳
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('scores')


if __name__ == "__main__":
    agent = Agent(scale=12)
    env = gym.make('CartPole-v1')
    # The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)

    '''
        # CartPole-v1环境的简单测试
        for i_episode in range(10):
        observation = env.reset()
        while True:
            env.render()
            print(observation)
            # print(agent.discrete(observation))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished")
                break
    '''

    score_record = []
    episode_num = 5000
    for i_episode in range(episode_num):
        state = env.reset()
        while True:
            action = agent.choose_action(state, epsilon=0.2)
            next_state, reward, done, info = env.step(action)
            # store the experience of this step
            # every-visit MC：一个episode内同一个state出现多次均用于q值更新；与之对应的是first-visit MC：每个episode只取第一次出现的state
            agent.store_experience_in_one_episode(state, action, reward)
            if done:
                agent.update_q_value()
                score = len(agent.episode_buffer)
                score_record.append(score)
                agent.clear_buffer()
                print(f"Score of episode {i_episode} is {score}.")
                break
            state = next_state
    env.close()
    agent.plot(episode_num, score_record, 'r')
    plt.show()

'''
# 保存训练参数至pkl文件 and 加载pkl文件存储的已训练参数至测试环境的agent上
import pickle
def save_parameters(agent):
    save_param = {'scale': agent.scale, 'explore': agent.explore, 'full_memory': agent.full_memory}
    with open('agent_params.pkl', 'wb') as f:
        pickle.dump(save_param, f)
def restore_parameters():
    with open('agent_params.pkl', 'rb') as f:
        save_param = pickle.load(f)
        agent = Agent(scale=save_param['scale'], explore=save_param['explore'])
        agent.full_memory = save_param['full_memory']
    return agent
# save_parameters(agent)
# agent2 = restore_parameters()
'''
