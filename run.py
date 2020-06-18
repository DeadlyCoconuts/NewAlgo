import gym
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import envs

from agent.agent import Agent
from agent.agent import Transition
from agent.objective_functions.multi_objective_optimisation import MultiObjectiveOpt
from agent.objective_functions.linear_objective import LinearObjectiveOpt
from algorithm.core import run_a2c_gtp
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic
from gym.wrappers import Monitor
from util import plot_target_evolution

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env_code = "maze-preset-3x3-kpi-v0"
env = gym.make(env_code).env
env.reset()

target = env.unwrapped.target
print(target)
print(env.unwrapped.v_mean)

agent = Agent(MultiObjectiveOpt(target), 500)

run_a2c_gtp(env, env_code, agent, max_time=10)
print(agent.avg_reward_list[-1])

transposed_reward_list = np.array(agent.avg_reward_list).T.tolist()

plot_target_evolution(target, transposed_reward_list, 10, env.unwrapped.num_rewards)
