import ctypes

import threading
from multiprocessing.sharedctypes import Array, RawArray, synchronized

from multiprocessing import Process, Manager
import torch
import numpy as np
from parallel_dqn.model import QNetwork
from utils.atomic_int import AtomicInteger


UPDATE_EVERY = 5
LR = 5e-4  # learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ParameterServer(Process):
    model = None
    optimizer = None


    def __init__(self, q, current_q, conn, state_size, action_size, seed, num_threads):
        self.q = q
        self.current_q = current_q
        self.conn = conn
        self.num_threads = num_threads

        self.time = 1

        self.model = QNetwork(state_size, action_size, seed).to(device)
      #  self.__class__.qs = qs
      #  self.__class__.num_threads = num_threads

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

     #   Array(ctypes.POINTER(type(ctypes.c_int)), 1)
       # i = RawArray('d', (9, 2, 8))
        #multiprocessing.Value('', ctypes.pointer(i))

        #
        # a = self.__class__.model.parameters()
        # self.shm = shared_memory.SharedMemory(create=True, size=a.nbytes, name='parameters')
        #
        # b = np.ndarray(a.shape, dtype=a.dtype, buffer=self.shm.buf)
        # b[:] = a[:]  # Copy the original data into shared memory
        #
        # self.__class__.shmtype = a.dtype
        # self.__class__.shmsize = a.shape

        super(ParameterServer, self).__init__()

    def run(self):
        grads = [None for _ in range(self.num_threads)]
        while True:
            i, msg = self.q.get()
            self.time += 1
            grads[i] = msg

            if self.time % UPDATE_EVERY == 0:
                self.apply_gradients(grads)
                #grads = []

    # def initialize_gradients(self, i, gradients):
    #     pass
    #   #  self.gradients[i] = gradients


    def apply_gradients(self, gradients):
       # if N.inc() % 10 == 0:
       # with lock:
         #   gradients = []

            # for q in cls.qs:
            #     while not q.empty():
            #         try:
            #             id, arr = q.get(False)
            #             gradients.append(arr)
            #         except:
            #             pass
      #  print(self.grads)
        gradients = [g for g in gradients if g is not None]
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()

        # for q in self.update_qs:
        #     q.put([p for p in self.model.parameters()])
        while not self.current_q.empty():
            try:
                self.current_q.get(False)
            except:
                pass

        self.current_q.put([p for p in self.model.parameters()])


    @classmethod
    def get_weights(cls):
        return cls.model.get_weights()

    # @classmethod
    # def get_parameters(cls):
    #     return cls.model.parameters()

    @classmethod
    def soft_update(cls, local_model, tau):
        #local_model = cls.recv_conn.recv()
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(cls.model.parameters(), local_model):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)