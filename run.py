import gym
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from agent.agent import Agent
from agent.agent import Transition
from agent.objective_functions.multi_objective_optimisation import MultiObjectiveOpt
from algorithm.core import run_A2C_GTP
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

"""
target = ScalarRewardEstimator(3, 3, 10).to(device)
target_optimizer = optim.Adam(target.parameters())
agent = Agent(MultiObjectiveOpt(np.array([0, 1, 2])), 10000)

values_a = np.random.rand(10000, 3)  # no * reward_dim
values_b = values_a.copy()[:9999, :]

grad = np.random.uniform(-1, 1, 3)
print(grad)


for i in range(10000 - 1):
    if i % 4 == 0:
        action = 0
    elif i % 4 == 1:
        action = 1
        values_b[i] = np.zeros(3)
    elif i % 4 == 2:
        action = 2
    elif i % 4 == 3:
        action = 3
    agent.add_transition(torch.tensor(values_a[i]).float(), torch.tensor([[action]]), torch.tensor(values_a[i+1]).float(),
                         torch.tensor(0).unsqueeze(0).float())

rewards = np.dot(values_b, grad)[:9999]

print(rewards)

batch = Transition(*zip(*agent.transition_list))
features_batch = torch.cat(batch.features).view(-1, 3)
action_batch = torch.cat(batch.action)
#print(action_batch)

for i in range(10000 - 1):
    scalar_reward_value = target(features_batch[i]).gather(0, action_batch[i])
    loss = (scalar_reward_value - torch.tensor(rewards[i]).float().unsqueeze(0).detach()).pow(2)
    #print(loss)
    target_optimizer.zero_grad()
    loss.backward()
    target_optimizer.step()

test = np.random.rand(3)
test_tensor = torch.tensor(test).float()

print(np.dot(test, grad))
with torch.no_grad():
    print(target(test_tensor))
"""
"""
scalar_reward_values = target(features_batch)
print(scalar_reward_values)
scalar_reward_values = scalar_reward_values.gather(1, action_batch)
print(scalar_reward_values)

loss = (scalar_reward_values - reward_batch).pow(2).sum()
target_optimizer.zero_grad()
loss.backward()
target_optimizer.step()

test = np.random.rand(3)
test_tensor = torch.tensor(test).float()

print(test_tensor)
print(np.dot(test, grad))
#print(torch.tensordot(test, torch.tensor(grad).float(), dims=1))
print(target(test_tensor))

"""
"""
for name, param in target.named_parameters():
    if param.requires_grad:
        print(name)
        print(param.data)
"""


model = ActorCritic(3, 3, 10).to(device)
values = (torch.rand(10000, 3))

policy_dist, value = model.forward(values[0])
print(policy_dist)


action = np.random.choice(3, p=policy_dist.detach().numpy())
b = torch.log(policy_dist[action])
print(action)
print(b)


"""
for i in range(10000):
    output = model(values[i])
    loss = (output - values_b[i]).pow(2).sum()
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()
"""
