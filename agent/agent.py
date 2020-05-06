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
        self.avg_reward_list.append(np.mean(self.reward_list, axis=0))
        self.objective_list.append(objective)
        self.feature_list.append(features)

    def get_objectives(self, w):
        return self.objective_function.objective(w), self.objective_function.grad_objective(w)
