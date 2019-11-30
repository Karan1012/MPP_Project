# Abstract class for an agent
from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def act(self, state, eps=0.):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def learn(self, experiences, gamma):
        pass
