# Distributed replay memory
import random
import threading
from collections import deque, namedtuple
from multiprocessing import Manager, Process

import torch
import numpy as np

from utils.atomic_int import AtomicInteger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 64  # minibatch size
#
# class Memory():
#     lock = threading.Lock()
#     experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#     memory = deque(maxlen=BUFFER_SIZE)
#
#     # Need to lock
#     @classmethod
#     def add(cls, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = cls.experience(state, action, reward, next_state, done)
#         with cls.lock:
#             cls.memory.append(e)
#
#     @classmethod
#     def sample(cls, device):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(cls.memory, k=BATCH_SIZE)
#
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
#             device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
#             device)
#
#         return (states, actions, rewards, next_states, dones)
#
#     @classmethod
#     def __len__(cls):
#         """Return the current size of internal memory."""
#         return len(cls.memory)
#
#     @classmethod
#     def get_batch_size(cls):
#         return BATCH_SIZE


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)

        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.shm = shared_memory.create_shared_memory((), num_processes)
        self.seed = random.seed(seed)
      #  self.N = AtomicInteger(-1)


    def add(self, state, action, reward, next_state, done):
        # e = (state, action, reward, next_state, done)
        # shared_memory.write_to_shared_memory(self.N.inc(), e, self.shm, self.experience)

       #  """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
       #
       #  if len(self.memory) >= self.buffer_size:
       #      self.memory.pop(0)
       #
       #
       #  self.memory.append(e)
       #
       #  #self.memory.append(e)

    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

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