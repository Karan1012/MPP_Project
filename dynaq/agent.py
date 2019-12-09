from multiprocessing import Lock

import torch
from torch import optim

from dynaq.env_model import EnvModelNetwork
from dynaq.model import QNetwork
from dynaq.worker import DynaQWorker, device
from dynaq.world_worker import DynaQWorldWorker
from parallel_dqn.parameter_server import ParameterServer
from parallel_dqn.replay_buffer import ReplayBuffer
from utils.agent import Agent

UPDATE_EVERY = 5

class DynaQAgent(Agent):
    def __init__(self, env, num_threads, gamma, lr, global_max_episode):
      #  self.ps = ParameterServer(env.observation_space.shape[0], env.action_space.n, 0, num_threads, UPDATE_EVERY, lr)
        # TODO: It would probably be better if this was shared between all processes
        # However, that might hurt performance
        # Could instead maybe create one global memory and send a few experiences in batches?
        #  self.memory = ReplayBuffer(env.action_space.n, BUFFER_SIZE, BATCH_SIZE)

      #  self.env_model = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)


        state_size, action_size = env.observation_space.shape[0], env.action_space.n

        self.world_model = EnvModelNetwork(state_size, action_size).to(device)
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=lr)

        self.qnetwork_global = QNetwork(state_size, action_size) #.to(device)
        self.qnetwork_global.share_memory()

        self.qnetwork_target = QNetwork(state_size, action_size) #.to(device)
        self.qnetwork_target.share_memory()

        self.q = [torch.multiprocessing.Queue() for _ in range(5)]

        self.lock = Lock()

     #   a, b = Pipe()

        self.real_worker = DynaQWorker(id=0, env=env, state_size=env.observation_space.shape[0],
                                          action_size=env.action_space.n, n_episodes=global_max_episode, lr=lr,
                                          gamma=gamma, update_every=UPDATE_EVERY,
                                       global_network=self.qnetwork_global, target_network=self.qnetwork_target, world_model=self.world_model, q=self.q)

        self.workers = [DynaQWorldWorker(id=id, env=env, state_size=env.observation_space.shape[0],
                                          action_size=env.action_space.n, n_episodes=global_max_episode, lr=lr,
                                          gamma=gamma, update_every=UPDATE_EVERY,
                                       global_network=self.qnetwork_global, target_network=self.qnetwork_target, world_model=self.world_model, q=self.q, lock=self.lock) for id in range(num_threads-1)]
    def train(self):
       # self.ps.start()


        [worker.start() for worker in self.workers]

        self.real_worker.run()

        [worker.join() for worker in self.workers]
