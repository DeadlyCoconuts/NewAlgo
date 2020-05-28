import numpy as np
from .objective_function import ObjectiveFunction


class LinearObjectiveOpt(ObjectiveFunction):
    def __init__(self, target):
        self.target = target
        self.num_dim = len(self.target)

    def objective(self, w):
        diff = w - self.target
        return np.sum(diff)

    def grad_objective(self, w):
        return np.ones(self.num_dim)
        #(1. / self.num_dim) * (np.ones(self.num_dim) - np.minimum((w - self.target), 0))