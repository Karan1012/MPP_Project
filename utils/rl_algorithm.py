# Abstract class for an agent
from abc import ABC, abstractmethod


class RLAlgorithm(ABC):

    @abstractmethod
    def train(self):
        pass


