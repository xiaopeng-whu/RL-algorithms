import numpy as np
import matplotlib.pyplot as plt


class K_Armed_Bandits():
    def __init__(self):
        self.action_space = [1, 2, 3]
        self.q = np.array([0.0, 0.0, 0.0])  # 这里v和q等价，反映摇一个臂对应的平均收益
        self.counts = np.array([0.0, 0.0, 0.0])     # 记录每个摇臂被摇到的次数
        self.accumulated_reward = 0
        self.accumulated_reward_history = []
        self.all_counts = 0     # 记录执行动作总次数

    def step(self, action):
        if action == 1:
            return np.random.normal(1, 1)   # 返回均值为1，方差为1的正态分布
        elif action == 2:
            return np.random.normal(2, 1)
        else:                               # action == 3
            return np.random.normal(3, 1)

    def choose_action(self, policy, **kwargs):
        if policy == 'e_greedy':
            if np.random.random() < kwargs['epsilon']:
                action = np.random.choice(self.action_space)
            else:
                action = np.argmax(self.q) + 1  # 找到最大值的索引+1
        if policy == 'ucb':
            if 0 in self.counts:    # 如果有还没有摇过的摇臂，则选择它（因为此时它的不确定度无限大）
                index = np.where(self.counts==0)[0] # np.where()输出满足条件的元素的坐标，以tuple给出，元素多少维就多少个tuple
                # 如np.where([0, 1, 0] == 0) = ([0,2],)
                # print(index)
                action = index[0] + 1
            else:
                c = kwargs['c_degree']
                ucb_value = self.q + c * np.sqrt(np.log(self.all_counts) / self.counts)
                action = np.argmax(ucb_value) + 1

        return action

    def train(self, policy, episodes, **kwargs):
        i = 0
        for i in range(episodes):
            if policy == 'e_greedy':
                action = self.choose_action(policy, epsilon=kwargs['epsilon'])
            if policy == 'ucb':
                action = self.choose_action(policy, c_degree=kwargs['c_degree'])
            reward = self.step(action)
            self.q[action-1] = (self.q[action-1]*self.counts[action-1]+reward) / (self.counts[action-1]+1)
            self.counts[action-1] += 1
            self.all_counts += 1
            self.accumulated_reward += reward
            self.accumulated_reward_history.append(self.accumulated_reward)
        # print(self.accumulated_reward)

    def reset(self):
        self.q = np.array([0.0, 0.0, 0.0])
        self.counts = np.array([0.0, 0.0, 0.0])
        self.all_counts = 0
        self.accumulated_reward = 0
        self.accumulated_reward_history = []

    def plot(self, colors, policy):
        plt.figure(1)
        episodes = len(self.accumulated_reward_history)
        x = [i for i in range(episodes)]
        plt.plot(x, self.accumulated_reward_history, colors, label=policy)
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('accumulated_reward_history')


if __name__ == "__main__":
    KAB = K_Armed_Bandits()
    KAB.train(policy='e_greedy', episodes=100, epsilon=0.1)
    KAB.plot('r', 'e_greedy')
    KAB.reset()
    KAB.train(policy='ucb', episodes=100, c_degree=0.5)
    KAB.plot('g', 'ucb')
    KAB.reset()
    plt.show()


