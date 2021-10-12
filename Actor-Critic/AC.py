'''
    REINFORCE方法具有高方差、更新不稳定、收敛慢的问题，且当累积回报为0时无法进行有效学习
    将对Q的估计从MonteCarlo采样得到的R替换为使用神经网络拟合的Q，即为Actor-Critic方法
    价值函数可以有多种形式表示，这里使用Q函数，即QAC算法
    DQN是对最优Q函数的近似，使用的是Q-Learning算法；而这里是对Qπ的近似，使用的是SARSA算法
    参考：https://github.com/yc930401/Actor-Critic-pytorch
        https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
    但网上的代码全是A2C，没有最naive的QAC（可能因为方差太大且比A2C还要复杂没什么意义）
    这里尝试复现一下加深理解，按照原算法叙述在每个timestep都进行更新，感觉不太对，但还是按这个思路写
    跑的效果也是很差（很有可能是代码写的问题），以后经验丰富了再回头修改吧
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state, action):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        # print("state:", state)
        # print("action:", action)
        # print("value:", value)
        # q_value = value[action].item()
        q_value = value[action].unsqueeze(0)    # 这里不能用item，否则无法进行梯度更新
        # print("q_value:", q_value)
        return q_value


def train(actor, critic, n_episodes):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        log_probs = []
        rewards = []    # 存储实际的reward值
        q_values = []     # 存储critic计算的q值
        masks = []      # next_state存在的掩码
        next_states = []

        for i in count():
            # env.render()
            state = torch.FloatTensor(state).to(device)
            # print(state)
            dist = actor(state)
            action = dist.sample()
            q_value = critic(state, action)
            # print(q_value)
            q_values.append(q_value)
            log_prob = dist.log_prob(action).unsqueeze(0)   # !!!
            log_probs.append(log_prob)

            next_state, reward, done, info = env.step(action.item())

            rewards.append(reward)
            masks.append(1-done)

            if not done:
                next_state = torch.FloatTensor(next_state).to(device)
                next_states.append(next_state)
                state = next_state
            else:
                next_state = torch.zeros_like(state)
                next_states.append(next_state)
                print('Episode: {}, Score: {}'.format(i_episode, i))
                break
        # print(q_values, len(q_values))
        # print(torch.cat(q_values))
        # print(rewards, len(rewards))
        # print(masks, len(masks))
        # print(next_states, len(next_states))
        # print(log_probs, len(log_probs))
        # print(torch.cat(log_probs))
        # print(log_probs[0].item())
        for j in range(len(next_states)):
            if masks[j]:    # 这样处理最后一个状态没有用到，肯定是有问题的...
                dist = actor(next_states[j])
                action = dist.sample()
                next_q_value = critic(next_states[j], action)
                td_target = rewards[j] + next_q_value
                td_error = td_target - q_values[j]
                optimizerA.zero_grad()
                optimizerC.zero_grad()
                actor_loss = -(log_probs[j].item() * q_values[j])
                actor_loss.backward(retain_graph=True)
                critic_loss = td_error * q_values[j]
                critic_loss.backward(retain_graph=True)
                optimizerA.step()
                optimizerC.step()
                # print(f"{j} finished")

    env.close()


if __name__ == "__main__":
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    train(actor, critic, n_episodes=1000)

