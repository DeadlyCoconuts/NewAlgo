import gym
import torch
import numpy as np
import torch.optim as optim

from torch_models.reward_vector_estimator import RewardVectorEstimator
from agent.agent import Agent
from agent.objective_functions.multi_objective_optimisation import MultiObjectiveOpt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


"""
model = RewardVectorEstimator(3, 3, 10).to(device)
model_optimizer = optim.Adam(model.parameters())

values_a = (torch.rand(10000, 3))
values_b = values_a.clone().detach()

for i in range(10000):
    output = model(values_a[i])
    loss = (output - values_b[i]).pow(2).sum()
    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()

test = torch.rand(3)

print(test)
print(model(test))
"""

"""
target = np.array([0, 1, 2])
agent = Agent(MultiObjectiveOpt(target))
"""




