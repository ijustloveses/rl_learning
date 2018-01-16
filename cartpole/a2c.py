# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym

"""
Hyper Parameters
"""
STATE_DIM = 4     # gym 中 catepole-v0 的物理状态
ACTION_DIM = 2    # 可能的动作 left/right
STEP = 2000       # 多少次更新
SAMPLE_NUM = 30   # 最多多少个 time-steps 就更新一次


"""
Actor - Policy model
given a state, return a policy which are the probabilities of possible actions
model is a 3 dense layers neural network, activated by relu and ended with a softmax
"""
class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out
        

"""
Critic - Value Function Approximation model
given a state, return value evaluation based on a specific policy
model is a 3 dense layers neural network, activated by relu
output a batch of scalers as value of the batched inputed states
"""
class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        

"""
跑一次 episode，单个 episode 最多跑 sample_nums 个 time-steps; 其实后面看到，其实 sample_nums 是多少步做一次模型更新
task 就是 gym 的 env，用于在 gym 中运行一步
"""
def run_episode_within_sample_nums(actor_network, task, sample_nums, value_network, init_state):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    # 运行 sample_nums 个 time-steps
    for j in range(sample_nums):
        states.append(state)
        # 加 [] 是为了把单个 state 作为一个 batch 传入 actor_network，因为 network 需要 batched input
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        # 根据 actor 网络结果所表示的动作几率，来采样下一步动作； [0] 同样就是取 batch 样本中的第一个，也是唯一一个结果
        action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        # 在 gym 中运行，采样新的状态和所得到的 reward，以及是否结束
        next_state, reward, done, _ = task.step(action)
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state

        if done:
            is_done = True
            state = task.reset()
            break

    # 如果到了 sample_nums 步之后还没有挂掉，那么计算 final_r；否则 final_r 就是初始值 0，表示已经挂掉了
    if not is_done:
        final_r = value_network(Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    # 如果挂掉了，那么 state 就是重新开始一局的初始状态；否则 state 就是当前这个 episode 的最后的状态
    # 那么，下个 episode 的初始状态会使用 state，也就是说，如果本 episode 没有挂掉的话，下个 episode 会继续本 episode 的状态往下走
    # 故此，其实 sample_nums 也可以理解为最多多少个 time-step 做一次模型更新
    return states, actions, rewards, final_r, state


"""
衰减化的长期奖励，而且是计算每个 time-step 上向后看的长期奖励
"""
def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    # 类似于 backward-view 的方式，来计算长期奖励
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    task = gym.make("CartPole-v0")
    init_state = task.reset()

    value_network = ValueNetwork(input_size=STATE_DIM, hidden_size=40, output_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=0.01)

    actor_network = ActorNetwork(STATE_DIM, 40, ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr=0.01)

    steps = []
    task_episodes = []
    test_results = []

    for step in range(STEP):
        states, actions, rewards, final_r, current_state = run_episode_within_sample_nums(actor_network, task, SAMPLE_NUMS, value_network, init_state)
        # 上一次的 roll out 返回下次 roll out 的初始状态
        # 或者是上次挂掉了，那么使用系统的新初始状态；要么上次没挂掉，那么使用上次的最后状态，也就是接着上次继续往下跑
        init_state = current_state
        # actions are in one-hot format
        action_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM))
        state_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))

        # 训练，
        actor_network_optim.zero_grad()
        # 把所有的状态扔到 action network 中，得到每一个状态下各种动作的几率
        log_softmax_actions = actor_network(state_var)
        # 同样，扔到 value network 中，得到每个状态下的价值 V(S)，这些值和动作无关，可以理解为 critic 对每个状态期望价值的预测
        vs = value_network(state_var).detach()
        # 于是，可以计算 Action-Value Functions 的估值；
        # 之所以是 Action-Value Function，是因为这些 discounted_reward 都是根据采样 episode 中根据采样出来的动作所得到的系统奖励计算的
        qs = Variable(torch.Tensor(discounted_reward(rewards, 0.99, final_r)))
        # 于是，Advantage Function 就可以得到了
        # 这个模型中，只创建了一个 critic 估值网络，用于估算给定状态的 Value Function V(S)
        # 并没有创建 Action-Value Function 的估值网络，也没有使用 TD-error 来代替 Advantage Function
        # 而是根据实际采样若干个 time-step，而根据 discounted reward 来作为 Action-Value Function
        advantages = qs - vs

        # 定义 loss function
        # actor 的 loss 就是 policy gradient；我们的目的是最大化 policy 下的期望目标函数，也即期望奖励
        # 由于已经采样出 sample_nums 个样本，故此这个期望奖励就是在这个样本上的平均奖励
        # 而奖励本身使用 Advantage Function 以减少方差
        # 这里 log_softmax_action 是每个状态下各个动作的概率，action_var 为实际采取动作的 one-hot 表达
        # 故此 torch.sum(log_softmax_action * action_var, 1) 就是实际采取动作的概率；再乘以 advantages，就得到了期望奖励
        # 取负号，以便使用 gradient descent 而不是 ascent
        actor_network_loss = - torch.mean(torch.sum(log_softmax_action * action_var, 1) * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # value network 目的是估计 Value Function，通常使用 MSE 就是差值平方来做 loss function
        value_network_optim.zero_grad()
        target_values = qs
        # 把所有的状态扔到 value network 中，得到每个状态下的价值 V(S)
        values = value_network(state_var)
        # 定义 MSE loss
        criterion = nn.MSELoss()
        # 我们希望 value network 预测出来的值能够趋近于实际采样中得到的真实奖励
        # 这里没有采用 TD(0) 之类的策略来更新 value network，而是采用实际采样的结果
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
        value_network_optim.step()


