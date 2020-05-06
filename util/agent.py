import numpy as np


class Agent:
    def __init__(self, objective_function):
        self.action_list = list()
        self.reward_list = list()
        self.avg_reward_list = list()
        self.objective_list = list()

        self.objective_function = objective_function
        # state_list?

    def update_history(self, action, reward, objective):
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.avg_reward_list.append(np.mean(self.reward_list, axis=0))
        self.objective_list.append(objective)