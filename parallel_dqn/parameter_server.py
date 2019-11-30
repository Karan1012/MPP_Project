import numpy as np
import torch
import torch.multiprocessing as mp

from parallel_dqn.model import QNetwork
from utils.shared_adam import SharedAdam

UPDATE_EVERY = 5

class SharedGradients:
    def __init__(self, num_threads, params):
        self.gradients = []
        arr = np.array([np.array(t) for t in [p.data for p in params]])

        for _ in range(num_threads):
            grads = [torch.tensor(arr[i]) for i in range(arr.size)]
            [g.share_memory_() for g in grads]
            self.gradients.append(grads)

    def update(self, i, grads):
        for tp, fp in zip(self.gradients[i], grads):
            tp.data.copy_(torch.tensor(fp).data)

    def get(self):
        return [g for g in self.gradients]


class ParameterServer(mp.Process):

    def __init__(self, state_size, action_size, seed, num_threads, lr):
        super(ParameterServer, self).__init__()

        self.global_steps = mp.Value('i', 0)

        self.model = QNetwork(state_size, action_size, seed)
        self.model.share_memory()

        self.optimizer = SharedAdam(self.model.parameters(), lr=lr)
        self.optimizer.share_memory()

        self.gradients = SharedGradients(num_threads, self.model.parameters())

    def record_gradients(self, i, gradients):
        with self.global_steps.get_lock():
            self.global_steps.value += 1

        self.gradients.update(i, gradients)

        if self.global_steps.value % UPDATE_EVERY == 0:
            self.apply_gradients()


    def apply_gradients(self):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*self.gradients.get())
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()


    def get_parameters(self):
       return self.model.parameters()

