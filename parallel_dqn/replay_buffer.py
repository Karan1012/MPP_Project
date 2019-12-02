# Distributed replay memory
import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch.multiprocessing import queue
import torch.multiprocessing as mp

from utils.device import device


#class ReplayBuffer(mp.Process):
# TODO: Readers / writers lock when making this global?
class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) #torch.tensor()#
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    #def add(self, q):
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # while not q.empty():
        #     x = q.get()
        #     x_clone = x #x.clone()
        #
        #     #  """Add a new experience to memory."""
        #     e = self.experience(x_clone.state, x_clone.action, x_clone.reward, x_clone.next_state, x_clone.done)
        #     self.memory.append(e)
        #     del x
        # # do somethings with x

    def sample(self, num):
        # perm = torch.randperm(tensor.size(0))
        # idx = perm[:k]
        # samples = torch.tensor[idx]

        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=num)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)