from multiprocessing import Lock

import torch

from dynaq.world_model import WorldModelNetwork
from dynaq.model import QNetwork
from dynaq.worker import DynaQWorker, device
from dynaq.world_worker import DynaQWorldWorker
from utils.agent import Agent

UPDATE_EVERY = 5

class DynaQAgent(Agent):
    def __init__(self, env, do_render, num_threads, gamma, lr, global_max_episode):

        state_size, action_size = env.observation_space.shape[0], env.action_space.n

        self.world_model = WorldModelNetwork(state_size, action_size)

        self.qnetwork_global = QNetwork(state_size, action_size)
        self.qnetwork_global.share_memory()

        self.qnetwork_target = QNetwork(state_size, action_size)
        self.qnetwork_target.share_memory()

        self.q = [torch.multiprocessing.Queue() for _ in range(5)]

        self.lock = Lock()

        self.real_worker = DynaQWorker(id=0, env=env, state_size=env.observation_space.shape[0],
                                          action_size=env.action_space.n, n_episodes=global_max_episode, lr=lr,
                                          gamma=gamma, update_every=UPDATE_EVERY,
                                       global_network=self.qnetwork_global, target_network=self.qnetwork_target, q=self.q)

        self.workers = [DynaQWorldWorker(id=id, env=env, state_size=env.observation_space.shape[0],
                                          action_size=env.action_space.n, n_episodes=global_max_episode, lr=lr,
                                          gamma=gamma, update_every=UPDATE_EVERY,
                                       global_network=self.qnetwork_global, target_network=self.qnetwork_target, world_model=self.world_model, q=self.q, lock=self.lock, num_threads=num_threads) for id in range(num_threads-1)]
    def train(self):
       # self.ps.start()


        [worker.start() for worker in self.workers]

        self.real_worker.run()

        [worker.join() for worker in self.workers]
