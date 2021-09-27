# 参考https://github.com/ShreeshaN/ReinforcementLearningTutorials
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        action = np.random.choice(env.action_space.n)
    else:
        action = np.argmax(Q[state])
    return action


def decay(epsilon):     # epsilon随时间增加探索率降低
    return 0.99 * epsilon


def sarsa(env, episodes_num, epsilon, gamma, alpha):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for i in range(episodes_num):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        epsilon = decay(epsilon)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * td_error
            if done:
                break
            action = next_action
            state = next_state
    return Q


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    '''
        o  o  o  o  o  o  o  o  o  o  o  o          0:up
        o  o  o  o  o  o  o  o  o  o  o  o          1:right
        o  o  o  o  o  o  o  o  o  o  o  o          2:down
        x  C  C  C  C  C  C  C  C  C  C  T          3:left
    '''
    print("Observation space: ", env.observation_space)     # Discrete(48)
    print("Action space: ", env.action_space)               # Discrete(4)
    # for i_episode in range(1):
    #     observation = env.reset()
    #     while True:
    #         env.render()
    #         print(observation)
    #         # print(agent.discrete(observation))
    #         action = env.action_space.sample()
    #         print(action)
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished")
    #             break
    Q = sarsa(env, episodes_num=5000, epsilon=0.1, gamma=1.0, alpha=0.01)
    policy = np.zeros(env.observation_space.n)
    for i in range(env.observation_space.n):
        if np.all(Q[i]):    # 如果该状态存在（即有对应的q值）
            policy[i] = np.argmax(Q[i])
        else:
            policy[i] = -1
    print(policy.reshape((4, 12)))

