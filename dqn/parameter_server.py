import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim

from dqn.model import QNetwork
from utils.shared_adam import SharedAdam
from utils.shared_asgd import SharedASGD

TAU = .2  # for soft update of target parameters

class SharedGradients:
    def __init__(self):
        self.gradients = [] # [None for _ in range(num_threads)]

    # def initialize_gradient(self, params):
    #     arr = np.array([np.array(t) for t in [p.data for p in params]])
    #
    #     for _ in range(num_threads):
    #         grads = [torch.tensor(arr[i]) for i in range(arr.size)]
    #         [g.share_memory_() for g in grads]
    #         self.gradients.append(grads)

    def initialize(self, gradients):
        grads = [gradients.clone().detach() for i in range(len(gradients))]
        [g.share_memory_() for g in grads] # Store gradients in shared memory
        self.gradients = grads

    def update(self, i, gradients):
        # for tp, fp in zip(self.gradients[i], grads):
        #     tp.data.copy_(torch.tensor(fp).data)
        for tp, fp in zip(self.gradients, gradients):
            tp.data.add_(torch.tensor(fp).data)


# TODO: Fix to use Queue instead for performance as suggested by torch multiprocessing
class ParameterServerShard(mp.Process):
    def __init__(self, params, visible_params, lr, q):
        super(ParameterServerShard, self).__init__()

       # self.batch = mp.Queue()

        # params = torch.nn.Parameter(params.clone().detach())
        # params.share_memory_() # Store gradients in shared memory
        self.parameters = params
        self.visible_params = visible_params
         # [params]

        self.optimizer = torch.optim.Adam([params], lr=lr)
        self.q = q

       # self.optimizer = SharedASGD(self.parameters, lr=lr, t0=0)
       # self.optimizer = SharedAdam([self.parameters], lr=lr)
      #  self.optimizer.share_memory()
        self.optimizer.zero_grad()

    def sum(self, grads):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*grads)
        ]
        return summed_gradients

    def run(self):
        while True:
            gradients = []
            try:
               # while len(gradients) <= 10:
                while not self.q.empty():
                    gradients.append(self.q.get(False))
                #print("too many items getting queued")

            except:
                if not len(gradients):
                    gradients.append(self.q.get())

            if not len(gradients):
                gradients.append(self.q.get())

            local_grads = [g.clone() for g in gradients]
            for g in gradients:
                del g
            self.update(sum(local_grads) / len(gradients))
            self.visible_params.data.copy_(self.parameters.data)



    def update(self, gradients):
        self.optimizer.zero_grad()
        self.parameters.grad = gradients # sum(grads)/len(grads)
        self.optimizer.step()



    # def set_gradients(self, gradients):
    #     self.parameters.grad = gradients
    #     # for g, p in zip(gradients, self.parameters[0]):
    #     #    if g is not None:
        #         p.grad = g

    # def get(self):
    #     return self.parameters

    def soft_update(self, local_model):
        tau = TAU
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
       # for target_param, local_param in zip(self.parameters[0], local_model):
        self.parameters.data.copy_(tau * local_model.data.detach() + (1.0 - tau) * self.parameters.data.detach())




class ParameterServer():

    def __init__(self, state_size, action_size, seed, num_threads, update_every, lr):
        super(ParameterServer, self).__init__()

        # TODO: Figure out how to just create a tensor that can do ASGD without the model needed
        self.model = QNetwork(state_size, action_size, seed) # TODO remove


        self.time = mp.Value('i', 0)

        p = [p for p in self.model.parameters()]
        # params = [torch.nn.Parameter(p[i].clone().detach()) for i in range(len(p))]
        # [g.share_memory_() for g in params]  # Store gradients in shared memory
        # self.parameters = params
        self.qs = [mp.Queue() for q in range(len(p))]
        self.shard_mem = [p[i].share_memory_() for i in range(len(p))]
        self.shards = [ParameterServerShard(p[i], self.shard_mem[i], lr, self.qs[i]) for i in range(len(p))]
        [shard.start() for shard in self.shards]

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
        #with self.time.get_lock():
        # TODO just create this as a tensor instead, it works better
        self.time.value += 1

        for q, g in zip(self.qs, gradients):
           # shard.update(g, self.time.value)
            q.put(torch.from_numpy(g).share_memory_())

    # def set_gradients(self, gradients):
    #     for g, p in zip(gradients, self.parameters):
    #         if g is not None:
    #             p.grad = torch.tensor(g)
    #            # p.grad = torch.from_numpy(g)

    def get(self):
         return self.shard_mem

