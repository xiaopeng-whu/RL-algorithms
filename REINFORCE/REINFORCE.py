import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque

'''
    REINFORCE, a.k.a VPG(Vanilla Policy Gradient)
        参考：https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_19_REINFORCE_Algorithm-OLD.ipynb
        https://github.com/lbarazza/VPG-PyTorch/blob/master/vpg.py
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define policy network
class policy_net(nn.Module):
    def __init__(self, state_size=4, hidden_size=16, action_size=2):
        super(policy_net, self).__init__()
        self.h = nn.Linear(state_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    # define forward pass with one hidden layer with ReLU activation and softmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)   # 关于F.softmax(dim)的解释：https://blog.csdn.net/xinjieyuan/article/details/105346381
        return x

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


if __name__ == "__main__":
    torch.manual_seed(0)  # set random seed
    env = gym.make("CartPole-v0")   # 连续状态空间（Box(4,)）、离散动作空间（Discrete(2)）
    threshold = env.spec.reward_threshold   # 195
    # print(threshold)
    policy = policy_net(env.observation_space.shape[0], 16, env.action_space.n).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    gamma = 1.0     # policy gradient方法常常设置γ=1.0
    n_episodes = 10000
    max_steps = 1000
    scores_deque = deque(maxlen=100)
    # render_rate = 100  # render every render_rate episodes
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_steps):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
            else:
                state = next_state  # 忘了改变state的值，捉了好久bug...
        scores_deque.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= threshold:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break


