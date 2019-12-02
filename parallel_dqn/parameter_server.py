import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim

from parallel_dqn.model import QNetwork
from utils.shared_adam import SharedAdam
from utils.shared_asgd import SharedASGD

TAU = .2  # for soft update of target parameters

class SharedGradients:
    def __init__(self, num_threads):
        self.gradients = [None for _ in range(num_threads)]

    # def initialize_gradient(self, params):
    #     arr = np.array([np.array(t) for t in [p.data for p in params]])
    #
    #     for _ in range(num_threads):
    #         grads = [torch.tensor(arr[i]) for i in range(arr.size)]
    #         [g.share_memory_() for g in grads]
    #         self.gradients.append(grads)

    def initialize(self, i, gradients):
        grads = [gradients[i].clone().detach() for i in range(len(gradients))]
        [g.share_memory_() for g in grads] # Store gradients in shared memory
        self.gradients[i] = grads

    def update(self, i, gradients):
        # for tp, fp in zip(self.gradients[i], grads):
        #     tp.data.copy_(torch.tensor(fp).data)
        for tp, fp in zip(self.gradients[i], gradients):
            tp.data.copy_(torch.tensor(fp).data)

    def sum(self):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*self.gradients)
        ]
        return summed_gradients

# TODO: Fix to use Queue instead for performance as suggested by torch multiprocessing
class ParameterServerShard:
    def __init__(self, params, lr):
        super(ParameterServerShard, self).__init__()

       # self.lock = mp.Lock()

        params = torch.nn.Parameter(params.clone().detach())
        params.share_memory_() # Store gradients in shared memory
        self.parameters = [params]

        self.optimizer = SharedASGD(self.parameters)
       # self.optimizer = SharedAdam(self.parameters, lr=lr)
        self.optimizer.share_memory()
        self.optimizer.zero_grad()

    # Need to fix with ASGD
    def record_gradients(self, gradients):
        self.set_gradients(gradients)
        self.optimizer.step()

        # self.lock.acquire()  # TODO this makes things slow
        # try:
        #     # self.optimizer.zero_grad()
        #     self.set_gradients(gradients)
        #     self.optimizer.step()
        # finally:
        #     self.lock.release()

    def update(self, gradients):
        self.optimizer.zero_grad()
        self.set_gradients(gradients)
        self.optimizer.step()

    def set_gradients(self, gradients):
        #for g, p in zip(gradients, self.parameters):
        #    if g is not None:
        self.parameters[0].grad = torch.tensor(gradients)

    def get(self):
        return self.parameters[0]


class ParameterServer(mp.Process):

    def __init__(self, state_size, action_size, seed, num_threads, update_every, lr):
        super(ParameterServer, self).__init__()

        # TODO: Figure out how to just create a tensor that can do ASGD without the model needed
        self.model = QNetwork(state_size, action_size, seed) # TODO remove
        #self.model.share_memory()

        p = [p for p in self.model.parameters()]
        # params = [torch.nn.Parameter(p[i].clone().detach()) for i in range(len(p))]
        # [g.share_memory_() for g in params]  # Store gradients in shared memory
        # self.parameters = params

        self.shards = [ParameterServerShard(p[i], lr) for i in range(len(p))]
        #[shard.start() for shard in self.shards]

        # TODO: Readers / Writers lock on gradients

    #    self.gradients = SharedGradients(num_threads)

    # def initialize_gradients(self, i, gradients):
    #     self.gradients.initialize(i, gradients)
    #
    # def apply_gradients(self):
    #     self.optimizer.zero_grad()
    #     self.model.set_gradients(self.gradients.sum())
    #     self.optimizer.step()

 # Need to fixwith ASGD
    # How to process asynchronously? Also do mini-batches
    def record_gradients(self, gradients):
        for shard, g in zip(self.shards, gradients):
            shard.update(g)

    # def set_gradients(self, gradients):
    #     for g, p in zip(gradients, self.parameters):
    #         if g is not None:
    #             p.grad = torch.tensor(g)
    #            # p.grad = torch.from_numpy(g)

    def get(self):
         return [shard.get() for shard in self.shards]

