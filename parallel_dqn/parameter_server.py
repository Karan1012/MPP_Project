from multiprocessing import Process

import numpy as np
import torch

from parallel_dqn.model import QNetwork

UPDATE_EVERY = 5
LR = 5e-4  # learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ParameterServer(Process):

    def __init__(self, q, current_q, conn, state_size, action_size, seed, num_threads):
        self.q = q
        self.current_q = current_q
        self.conn = conn
        self.num_threads = num_threads

        self.time = 1

        self.model = QNetwork(state_size, action_size, seed).to(device)
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
