# Abstract class for an agent
from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def train(self):
        pass


