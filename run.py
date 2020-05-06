import gym
import math
import torch
import numpy as np
import torch.optim as optim

from agent.agent import Agent
from agent.objective_functions.multi_objective_optimisation import MultiObjectiveOpt
from algorithm.core import run_A2C_GTP
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



model = RewardVectorEstimator(3, 3, 3).to(device)
target = ScalarRewardEstimator(3, 3, 3).to(device)

target.eval()

model_optimizer = optim.Adam(model.parameters())

values_a = (torch.rand(10000, 1, 3))  # no * actions * reward_dim
values_b = values_a.clone().detach()

for i in range(10000):
    output = model(values_a[i])
    loss = (output - values_b[i]).pow(2).sum()
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()

state_dict = model.state_dict()
grad = torch.rand(1, 3)
target.add_grad_objective_weights(grad, state_dict)
test = torch.rand(1, 3)

print(test)
print(model(test))
print(target(test))

for name, param in target.named_parameters():
    if param.requires_grad:
        print(name)
        print(param.data)

"""
target = np.array([0, 1, 2])

env = gym.envs.make('CartPole-v1')
agent = Agent(MultiObjectiveOpt(target))

#run_A2C_GTP(env, agent)
"""

"""
model = ActorCritic(3, 3, 10).to(device)
values = (torch.rand(10000, 3))

policy_dist, value = model.forward(values[0])
print(policy_dist)

policy_dist = policy_dist.detach().numpy()

action = np.random.choice(3, p=policy_dist)

print(action)
"""

"""
for i in range(10000):
    output = model(values[i])
    loss = (output - values_b[i]).pow(2).sum()
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()
"""
