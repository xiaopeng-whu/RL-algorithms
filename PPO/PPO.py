'''
    TRPO computes the gradients with a complex second-order method. 网上资源很少，应该是PPO在各方面都优于TRPO就没有必要复现流程复杂的TRPO了。
    目前，PPO是强化学习领域最流行、热门的算法之一。
        在工程问题中，人们一般的直觉是——比较小的问题使用DQN作为Baseline（收敛较快、便于训练），而比较大、比较复杂的问题使用PPO作为Baseline（效果较好）。
    PPO: Proximal Policy Optimization Algorithms
        PPO算法主要有两个变种，一个是结合KL penalty的PPO-penalty，一个是用了clip方法的PPO-clip，这里实现的是后者。
    参考：https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
'''