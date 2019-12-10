import torch
import torch.multiprocessing as mp
from torch import optim

from a3c.model import ActorCriticNetwork

from a3c.worker import A3CWorker
from utils.agent import Agent


class A3CAgent(Agent):

    def __init__(self, env, do_render, num_threads, gamma, lr, global_max_episode):
        self.env = env
        state_size, action_size = env.observation_space.shape[0], env.action_space.n

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.global_network = ActorCriticNetwork(state_size, action_size)
        self.global_network.share_memory()
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr)

        self.workers = [A3CWorker(i, env, state_size, action_size, self.gamma, lr, self.global_network, self.global_optimizer,
                                        self.global_episode,  self.GLOBAL_MAX_EPISODE) for i in range(num_threads)]

    def train(self):
        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
