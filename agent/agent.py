import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('features', 'action', 'next_features', 'reward'))  # TODO: Do I really need next features?

class Agent:
    def __init__(self, objective_function, max_memory_capacity):
        self.action_list = list()
        self.reward_list = list()
        self.avg_reward_list = list()
        self.objective_list = list()
        self.feature_list = list()

        self.transition_list = list()
        self.max_memory_capacity = max_memory_capacity
        self.transition_counter = 0

        self.objective_function = objective_function

    def update_history(self, features, action, reward_vector):
        self.feature_list.append(features)
        self.action_list.append(action)
        self.reward_list.append(reward_vector)

        # Objective function data
        new_avg_reward = np.mean(self.reward_list, axis=0)
        self.avg_reward_list.append(new_avg_reward)
        self.objective_list.append(self.objective_function.objective(new_avg_reward))

    def add_transition(self, *args):
        if len(self.transition_list) < self.max_memory_capacity:
            self.transition_list.append(None)
        self.transition_list[self.transition_counter] = Transition(*args)
        self.transition_counter = (self.transition_counter + 1) % self.max_memory_capacity

    def get_objectives(self):
        if not self.avg_reward_list:
            current_avg_reward = 0
        else:
            current_avg_reward = self.avg_reward_list[-1]
        return self.objective_function.objective(current_avg_reward),\
               self.objective_function.grad_objective(current_avg_reward)
