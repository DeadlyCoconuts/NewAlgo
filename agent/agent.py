import numpy as np


class Agent:
    def __init__(self, objective_function):
        self.action_list = list()
        self.reward_list = list()
        self.avg_reward_list = list()
        self.objective_list = list()
        self.feature_list = list()

        self.objective_function = objective_function

    def update_history(self, features, action, reward, objective):
        self.action_list.append(action)
        self.reward_list.append(reward)

        new_avg_reward = np.mean(self.reward_list, axis=0)
        self.avg_reward_list.append(new_avg_reward)

        self.objective_list.append(self.objective_function(new_avg_reward))
        self.feature_list.append(features)

    def get_objectives(self):
        current_avg_reward = self.avg_reward_list[-1]
        return self.objective_function.objective(current_avg_reward),\
               self.objective_function.grad_objective(current_avg_reward)
