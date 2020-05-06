from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def objective(self):
        pass

    @abstractmethod
    def grad_objective(self):
        pass

    @abstractmethod
    def objective_dim(self):
        pass
