import ctypes
import multiprocessing
import threading
from multiprocessing.sharedctypes import Array, RawArray
from multiprocessing import Process, Manager
import torch
import numpy as np

from parallel_dqn.model import QNetwork
from utils.atomic_int import AtomicInteger


UPDATE_EVERY = 10
LR = 5e-4  # learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N = AtomicInteger(0)
lock = threading.Lock()

class ParameterServer(object):
    model = None
    optimizer = None

    def __init__(self, state_size, action_size, seed, qs, num_threads):
        self.__class__.qs = qs
        self.__class__.num_threads = num_threads
        self.__class__.model = QNetwork(state_size, action_size, seed).to(device)
        self.__class__.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR)

    def initialize_gradients(self, i, gradients):
        pass
      #  self.gradients[i] = gradients

    @classmethod
    def apply_gradients(cls):
        if N.inc() % 10 == 0:
            with lock:
                gradients = []

                for q in cls.qs:
                    while not q.empty():
                        try:
                            id, arr = q.get(False)
                            gradients.append(arr)
                        except:
                            pass

                gradients = [g for g in gradients if g is not None]
                if len(gradients):
                    summed_gradients = [
                        np.stack(gradient_zip).sum(axis=0)
                        for gradient_zip in zip(*gradients)
                    ]
                    cls.optimizer.zero_grad()
                    cls.model.set_gradients(summed_gradients)
                    cls.optimizer.step()


    @classmethod
    def get_weights(cls):
        return cls.model.get_weights()

    @classmethod
    def get_parameters(cls):
        return cls.model.parameters()