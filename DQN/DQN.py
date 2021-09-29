import gym
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

'''
Typically, people implement replay buffers with one of the following three data structures:
    collections.deque: (try first)
        deque is very easy to handle once you initialize its maximum length (e.g. deque(maxlen=buffer_size)). 
        However, the indexing operation of deque gets terribly slow as it grows up because it is internally doubly linked list.
    list: 
        list is an array, so it is relatively faster than deque when you sample batches at every step. Its amortized cost of Get item is O(1).
    numpy.ndarray: 
        numpy.ndarray is even faster than list due to the fact that it is a homogeneous array of fixed-size items, 
        so you can get the benefits of locality of reference. Whereas list is an array of pointers to objects, even when all of them are of the same type. 
'''
Transition = collections.namedtuple("experience", ("state", "action", "reward", "done", "next_state"))
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indexs = np.random.choice(len(self.buffer), batch_size, replace=False)
        # 序列解包 https://www.jianshu.com/p/f9e7140ce19d
        states, actions, rewards, dones, next_states = zip(*[self.buffer[i] for i in indexs])
        # 返回numpy数组为了后续计算方便
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    def __len__(self):  # 在我们自定义的类中，要让len()函数工作正常，类必须提供一个特殊方法__len__()，这样就可以len(buffer)得到缓冲区当前长度了
        return len(self.buffer)

# the model in Nature has three convolution layers followed by two fully connected layers.
# All layers are separated by rectified linear unit(ReLU) nonlinearities.
class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super(DQN, self).__init__()     # 子类Network继承父类nn.Module的所有属性和方法，并用父类方法初始化
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),    # in_channels：输入图片的层数，一般为RGB的3层
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # 传递给全连接层的卷积层的输出必须在全连接层接受输入之前进行flatten
        # PyTorch没有可以把一个3D tensor transform为1D tensor的 "flatter" layer
        # conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(64*22*16, 512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )

    # # 我们不知道卷积层输出值的具体数目，为了不用硬编码，利用输入shape构造一个fake tensor输入到卷积层中
    # def _get_conv_out(self, shape): # 模型创建时即被调用，所以很快
    #     o = self.conv(torch.zeros(1, *shape))
    #     return int(np.prod(o.size()))   # 返回结果等同于参数数目，

    def forward(self, x):   # 接受4D的输入tensor:(batch_size, color_channel, height, width)
        x = x.float() / 255     # 要将图像数据归一化，否则会报错
        # print(x.size())  # (1, 3, 210, 160)
        # print(self.conv(x).size())  # 要确定卷积层输出维度，这里先在全连接层上硬编码，后面考虑自动计算的方式 # (1, 64, 22, 16)
        conv_out = self.conv(x).view(x.size()[0], -1)   # reshape为(batch_size, *)   # (1, 64*22*16)
        # print(conv_out.size())
        return self.fc(conv_out)


class Agent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99           # 折扣因子
        self.epsilon = 1            # 初始epsilon值
        self.epsilon_decay = .995   # 每次随机衰减0.005
        self.epsilon_min = 0.02     # epsilon衰减最小值后不再衰减
        self.lr = 1e-4              # 学习率
        self.target_net_update_frequency = 10000    # target network的更新频率
        self.batch_size = 32
        self.buffer_capacity = 10000
        self.buffer = ExperienceBuffer(capacity=self.buffer_capacity)
        self.policy_net = DQN(in_channels=env.observation_space.shape[2], n_actions=env.action_space.n)
        self.target_net = DQN(in_channels=env.observation_space.shape[2], n_actions=env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始化两个网络参数相同
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def choose_action(self, state):
        global steps_done
        eps = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        steps_done += 1
        if np.random.random() < eps:    # random
            # return self.env.action_space.sample()   # 如：2
            return torch.tensor(np.random.randint(self.env.action_space.n))  # 如：tensor(2)
        else:
            q_values = self.policy_net(state)
            return q_values.max(1)[1]               # 如：tensor([2])  格式不统一？有的代码后面为什么加.view(1,1)？

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
        # print("states:", states)

        return

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))  # (210, 160, 3) 转置为(3, 210, 160)
        # print(state.shape)
        state = torch.from_numpy(state)     # numpy转为Tensor
        # print(state.unsqueeze(0).shape)
        return state.unsqueeze(0)   # 第0维增加一个维度作为batch_size->(1, 3, 210, 160)

    def train(self, episodes_num, max_steps, render=False):
        for i in range(episodes_num):
            obs = self.env.reset()
            state = self.get_state(obs)
            score = 0.0
            for t in range(max_steps):  # max_steps应能保证到达done或接近，太小（500）则始终无法done
                # print("state:", state)
                action = self.choose_action(state)
                # print("action:", action)
                if render:
                    self.env.render()
                observation, reward, done, info = self.env.step(action)
                score += reward
                if not done:
                    next_state = self.get_state(observation)
                else:
                    next_state = None
                # store the experience
                experience = Transition(state, action, reward, done, next_state)
                self.buffer.push(experience)
                state = next_state
                # if steps_done > 100:
                #     self.optimize_model()
                if done:
                    print(f'Episode {i} is done.')
                    break
            if i % 20 == 0:     # 每20个episodes打印一次当前episode的得分
                print(f'Score is {score} during Episode {i}. ')

        return

    def plot(self):
        return


if __name__ == "__main__":
    '''
        Test Env:
            BreakoutNoFrameskip-v4：打砖块游戏
            PongNoFrameskip-v4：乒乓球游戏
            v0和v4的区别：带有v0的env表示会有25%的概率执行上一个action，而v4表示只执行agent给出的action，不会重复之前的action。
            带有Deterministic的env表示固定跳4帧，否则跳帧数随机从(2, 5)中采样。（(同一个动作在k帧中保持,这种设置可以为避免训练出的智能体超出人类的反应速率)）
            带有NoFrameskip的env表示没有跳帧。
    '''
    # 面对CartPole问题，可以进一步简化：
    #   1.无需预处理Preprocessing。也就是直接获取观察Observation作为状态state输入。
    #   2.只使用最基本的MLP神经网络，而不使用卷积神经网络。
    # env = gym.make('PongNoFrameskip-v4')
    # print(env.observation_space.shape, env.action_space.n)  # Box 对象不能用 env.observation_space.n
    # for i_episode in range(10):
    #     observation = env.reset()
    #     while True:
    #         env.render()
    #         # print(observation)
    #         # print(agent.discrete(observation))
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print(f"Episode {i_episode} finished")
    #             break
    # env.close()

    '''
        Test ExperienceBuffer
    '''
    # buffer = ExperienceBuffer(5)
    # exp1 = Transition(1, 2, 3, 4, 5)
    # buffer.push(exp1)
    # exp2 = buffer.sample(1)
    # print("exp2:",exp2)
    # print(len(buffer))

    '''
        Test DQN Network
    '''
    # env = gym.make('PongNoFrameskip-v4')
    # env.seed(1)  # 可选，设置随机数，以便让过程重现
    # env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
    # # 在gym中连续空间为Box，离散空间为Discrete
    # # 连续空间（Box）下，例如绝大多数的观测空间和部分环境的动作空间，通常使用env.observation_space.shape[0]来获得连续空间的维度
    # # 而离散空间（Discrete）需要使用env.action_space.n获得离散空间的维度；
    # print(env.observation_space, env.action_space)
    # # Good general-purpose agents don't need to know the semantics of the observations:
    # #   they can learn how to map observations to actions to maximize reward without any prior knowledge.
    # input_shape = env.observation_space.shape   # 对于atari游戏，观测值为(210, 160, 3)的三维图像值
    # print(input_shape, input_shape[0])
    # agent = Agent(env)
    # x = env.reset()
    # # print(x)
    # # print(x.shape)  # （210,160,3)
    # x = agent.get_state(x)
    # value = agent.policy_net(x)
    # print(value)
    # print(value.max(1))
    # print(value.max(1)[1])
    # print(value.max(1)[1].view(1,1))
    # print(agent.env.action_space.sample())
    # print(torch.tensor(np.random.randint(6)))

    '''
        Test self.train()
    '''
    env = gym.make('PongNoFrameskip-v4')
    # env.seed(1)  # 可选，设置随机数，以便让过程重现
    # env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
    steps_done = 0
    agent = Agent(env)
    agent.train(21, 5000)


