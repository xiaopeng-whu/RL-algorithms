'''
    A3C: Asynchronous Advantage Actor-Critic
    并行使用多个环境能够提高训练的稳定性（降低方差），PG方法是在当前策略上进行更新，故没办法使用DQN的replay buffer来保证i.i.d假设
        使用并行环境就可以避免从单一episode的样本中更新导致的不稳定，但依然存在sample efficiency的问题
    严格意义上A3C的通信不是并行的而是串行的
    可以有两种方式实现并行：1.数据并行（样本收集到一起计算loss） 2.梯度并行：各自计算gradient再收集到一起进行SGD更新
        前者更容易实现，后者在GPU资源充足的情况下更好，单个GPU两者性能相似。
    参考：https://github.com/seungeunrho/minimalRL/blob/master/a3c.py
'''
import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch.multiprocessing as mp
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# hyperparameters
n_train_processes = 3
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 1000
update_interval = 5


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist
#
#
# def a2c(env):
#     num_inputs = env.observation_space.shape[0]
#     num_outputs = env.action_space.n
#
#     actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
#     ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
#
#     all_lengths = []
#     average_lengths = []
#     all_rewards = []
#     # entropy_term = 0    # 熵项是如何处理的？这个不应该是每次episode开始都清零吗？
#
#     for episode in range(max_episodes):
#         entropy_term = 0
#         log_probs = []
#         values = []  # 存储V值
#         rewards = []
#
#         state = env.reset()
#         for steps in range(num_steps):
#             value, policy_dist = actor_critic(state)
#             value = value.detach().numpy()[0, 0]
#             dist = policy_dist.detach().numpy()
#
#             action = np.random.choice(num_outputs, p=np.squeeze(dist))
#             log_prob = torch.log(policy_dist.squeeze(0)[action])
#
#             # entropy = -np.sum(np.mean(dist) * np.log(dist))   # 这个不是按信息熵的定义吧？
#             entropy = -sum(dist[0][i] * np.log(dist[0][i]) for i in range(len(dist[0])))
#
#             new_state, reward, done, _ = env.step(action)
#
#             rewards.append(reward)
#             values.append(value)
#             log_probs.append(log_prob)
#             entropy_term += entropy
#             state = new_state
#
#             if done or steps == num_steps - 1:
#                 Qval, _ = actor_critic(new_state)   # 即使done，next_state也可以存在并计算V值，理想情况下此时V值为0
#                 Qval = Qval.detach().numpy()[0, 0]
#                 all_rewards.append(np.sum(rewards))
#                 all_lengths.append(steps)
#                 average_lengths.append(np.mean(all_lengths[-10:]))  # 按最新10次试验的平均length计算
#                 if episode % 10 == 0:
#                     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format
#                                      (episode, np.sum(rewards), steps, average_lengths[-1]))
#                 break
#
#         # compute Q values
#         Qvals = np.zeros_like(values)
#         for t in reversed(range(len(rewards))):
#             Qval = rewards[t] + GAMMA * Qval
#             Qvals[t] = Qval
#
#         # update actor critic
#         values = torch.FloatTensor(values)
#         Qvals = torch.FloatTensor(Qvals)
#         log_probs = torch.stack(log_probs)
#
#         advantage = Qvals - values
#         actor_loss = (-log_probs * advantage).mean()
#         critic_loss = 0.5 * advantage.pow(2).mean()     # MSE
#         # print(actor_loss)
#         # print(critic_loss)
#         # print(entropy_term)
#         ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
#
#         ac_optimizer.zero_grad()
#         ac_loss.backward()  # 两个网络按一个AC网络实现的，这样更新和分开两个网络有区别吗？
#         ac_optimizer.step()
#
#     # Plot results
#     smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()     # 滑动窗口平均让曲线更平滑
#     smoothed_rewards = [elem for elem in smoothed_rewards]
#     plt.plot(all_rewards)
#     plt.plot(smoothed_rewards)
#     plt.plot()
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.show()
#
#     plt.plot(all_lengths)
#     plt.plot(average_lengths)
#     plt.xlabel('Episode')
#     plt.ylabel('Episode length')
#     plt.show()


def train(global_model, rank):
    local_model = ActorCritic(4, 2, hidden_size)
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make('CartPole-v1')
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    all_lengths = []
    average_lengths = []
    all_rewards = []
    # entropy_term = 0    # 熵项是如何处理的？这个不应该是每次episode开始都清零吗？

    for episode in range(max_episodes):
        entropy_term = 0
        log_probs = []
        values = []  # 存储V值
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = local_model(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])

            # entropy = -np.sum(np.mean(dist) * np.log(dist))   # 这个不是按信息熵的定义吧？
            entropy = -sum(dist[0][i] * np.log(dist[0][i]) for i in range(len(dist[0])))

            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == num_steps - 1:
                Qval, _ = local_model(new_state)  # 即使done，next_state也可以存在并计算V值，理想情况下此时V值为0
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))  # 按最新10次试验的平均length计算
                # if episode % 10 == 0:
                #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format
                #                      (episode, np.sum(rewards), steps, average_lengths[-1]))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()  # MSE

        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        optimizer.zero_grad()
        ac_loss.backward()  # 两个网络按一个AC网络实现的，这样更新和分开两个网络有区别吗？

        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def _test(global_model):
    env = gym.make('CartPole-v1')
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    score = 0.0
    print_interval = 20

    for n_epi in range(max_episodes):
        done = False
        state = env.reset()
        while not done:
            value, policy_dist = global_model(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))

            new_state, reward, done, _ = env.step(action)
            state = new_state
            score += reward

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()


if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    # a2c(env)

    global_model = ActorCritic(4, 2, hidden_size)
    global_model.share_memory()
    # 多进程编程利用了计算机多核CPU的优势，实现真正的并行计算
    processes = []
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=_test, args=(global_model,))
        else:
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()  # 开始执行子进程
        processes.append(p)
    for p in processes:
        p.join()  # 等待子进程结束