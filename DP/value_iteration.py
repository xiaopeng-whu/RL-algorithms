import numpy as np
import gym
import gym.spaces as spaces
import time

'''
    FrozenLake-v0:
        SFFF
        FHFH
        FFFH
        HFFG
    S: initial state
    F: frozen lake
    H: hole
    G: the goal
    Red square: indicates the current position of the player
'''
# action mapping for display the final result
action_mapping = {
    3: '\u2191',    # UP
    2: '\u2192',    # RIGHT
    1: '\u2193',    # DOWN
    0: '\u2190'     # LEFT
}
print("0 1 2 3:")
print(' '.join([action_mapping[i] for i in range(4)]))


def play_episodes(environment, n_episodes, policy, discount_factor=0.999, random=False):
    """
    This function plays the given number of episodes given by following a policy or sample randomly from action_space.

    Parameters:
        environment: openAI GYM object
        n_episodes: number of episodes to run
        policy: Policy to follow while playing an episode
        random: Flag for taking random actions. If True no policy would be followed and action will be taken randomly

    Return:
        wins: Total number of wins playing n_episodes
        total_reward: Total reward of n_episodes
        avg_reward: Average reward of n_episodes

    """
    # initialize wins and total reward
    win_times = 0  # 到达goal的次数
    total_reward = 0

    # loop over number of episodes to play
    for episode in range(n_episodes):
        print("episode ", episode+1, ":")
        state = environment.reset()  # reset the environment every time when playing a new episode
        step_idx = 0  # 在一个episode的步的序号
        while True:
            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = environment.action_space.sample()
            else:
                action = policy[state]
            # take the next step
            next_state, reward, done, info = environment.step(action)
            # print(info)
            environment.render()
            total_reward += (discount_factor ** step_idx) * reward  # accumulate total reward with discount factor
            state = next_state  # change the state
            if done:
                if reward == 1.0:  # 如果因正常到达goal结束，则win
                    win_times += 1
                break
    average_reward = total_reward / n_episodes  # calculate average reward

    return win_times, total_reward, average_reward


def calc_action_value(env, state, V, discount_factor=0.99):
    """
    Function to  calculate action-state-value function

    Arguments:
        env: openAI GYM environment object
        state: state to consider
        V: Estimated Value for each state. Vector of length nS
        discount_factor: MDP discount factor

    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """

    # initialize vector of action values
    action_values = np.zeros(env.action_space.n)    # env.nA = env.action_space.n

    # loop over the actions we can take in an environment
    for action in range(env.action_space.n):
        # loop over the P_sa distribution.
        for probability, next_state, reward, info in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            action_values[action] += probability * (reward + (discount_factor * V[next_state]))

    return action_values


def update_policy(env, policy, V, discount_factor):
    """
    Helper function to update a given policy based on given value function.

    Arguments:
        env: openAI GYM environment object.
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """

    for state in range(env.nS):
        # for a given state compute state-action value.
        action_values = calc_action_value(env, state, V, discount_factor)

        # choose the action which maximizes the state-action value.
        policy[state] = np.argmax(action_values)

    return policy


def value_iteration(env, discount_factor=0.999, max_iteration=1000):
    """
    Algorithm to solve MDP.

    Arguments:
        env: openAI GYM environment object.
        discount_factor: MDP discount factor.
        max_iteration: Maximum No.  of iterations to run.

    Return:
        V: Optimal state-Value function. Vector of lenth nS.
        optimal_policy: Optimal policy. Vector of length nS.

    """
    # initialize value function
    V = np.zeros(env.observation_space.n)

    # iterate over max_iterations
    for i in range(max_iteration):
        #  keep track of change with previous value function
        prev_v = np.copy(V)

        # loop over all states
        for state in range(env.observation_space.n):
            # Asynchronously update the state-action value
            # action_values = calc_action_value(env, state, V, discount_factor)

            # Synchronously update the state-action value
            action_values = calc_action_value(env, state, prev_v, discount_factor)

            # select best action to perform based on highest state-action value
            best_action_value = np.max(action_values)
            # update the current state-value fucntion
            V[state] = best_action_value

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if np.all(np.isclose(V, prev_v)):   # np.isclose return array; np.allclose return bool
                print('Value converged at iteration %d' % (i + 1))
                break

    # initialize optimal policy
    optimal_policy = np.zeros(env.observation_space.n, dtype='int8')
    # update the optimal policy according to optimal value function 'V'
    optimal_policy = update_policy(env, optimal_policy, V, discount_factor)

    return V, optimal_policy


if __name__ == "__main__":
    # environment = gym.make('FrozenLake-v0')
    environment = gym.make("FrozenLake-v0", is_slippery=False)  # 关闭光滑表面设定的环境设置
    tic = time.time()
    opt_V, opt_policy = value_iteration(environment.env, max_iteration=1000)
    toc = time.time()
    elapsed_time = (toc - tic) * 1000
    print(f"Time to converge: {elapsed_time: 0.3} ms")
    print('Optimal Value function: ')
    print(opt_V.reshape((4, 4)))
    print('Final Policy: ')
    # print(opt_Policy)
    # print(' '.join([action_mapping[int(action)] for action in opt_Policy]))
    print(opt_policy.reshape((4, 4)))   # reshape没有实际改变原数组
    print(' '.join([action_mapping[action] for action in opt_policy[0:4]]))
    print(' '.join([action_mapping[action] for action in opt_policy[4:8]]))
    print(' '.join([action_mapping[action] for action in opt_policy[8:12]]))
    print(' '.join([action_mapping[action] for action in opt_policy[12:16]]))

    n_episode = 10
    win_times, total_reward, avg_reward = play_episodes(environment, n_episode, opt_policy, discount_factor=0.999,random=False)

    print(f'Total win_times with value iteration: {win_times}')
    print(f"Average rewards with value iteration: {avg_reward}")