from abc import ABC

import numpy as np
from .objective_function import ObjectiveFunction


class MultiObjectiveOpt(ObjectiveFunction):
    def __init__(self, target):
        self.target = target
        self.num_dim = len(self.target)

    def objective(self, w):
        diff = w - self.target
        return - (0.5 / self.num_dim) * np.dot(diff, diff) + (1. / self.num_dim) * np.dot(np.ones(self.num_dim), w)

    def grad_objective(self, w):
        return (1. / self.num_dim) * (self.target - w) + (1. / self.num_dim) * np.ones(self.num_dim)

    def objective_dim(self):
        return self.num_dim
