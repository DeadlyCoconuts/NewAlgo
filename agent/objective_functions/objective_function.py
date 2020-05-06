from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def objective(self, w):
        return

    @abstractmethod
    def grad_objective(self, w):
        return
