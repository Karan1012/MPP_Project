from multiprocessing import Lock

import torch

from dynaq.world_model import WorldModelNetwork
from dynaq.model import QNetwork
from dynaq.agent import DynaQAgent
from dynaq.world_agent import DynaQWorldAgent
from utils.rl_algorithm import RLAlgorithm


class DynaQ(RLAlgorithm):
    def __init__(self, env, do_render, num_threads, gamma, lr, global_max_episode):

        state_size, action_size = env.observation_space.shape[0], env.action_space.n

        self.world_model = WorldModelNetwork(state_size, action_size)

        self.qnetwork_global = QNetwork(state_size, action_size)
        self.qnetwork_global.share_memory()

        self.qnetwork_target = QNetwork(state_size, action_size)
        self.qnetwork_target.share_memory()

        self.q = [torch.multiprocessing.Queue() for _ in range(5)]
        self.lock = Lock()

        self.real_agent = DynaQAgent(id=0, env=env, state_size=state_size, action_size=action_size, n_episodes=global_max_episode, lr=lr,gamma=gamma,
                                       global_network=self.qnetwork_global, target_network=self.qnetwork_target, q=self.q)

        self.world_agents = [DynaQWorldAgent(id=id, state_size=state_size,action_size=action_size, n_episodes=global_max_episode, lr=lr,gamma=gamma,
                                       global_network=self.qnetwork_global, target_network=self.qnetwork_target, world_model=self.world_model,
                                        q=self.q, lock=self.lock, num_threads=num_threads) for id in range(num_threads-1)]
    def train(self):

        [agent.start() for agent in self.world_agents]

        self.real_agent.run()

        [agent.join() for agent in self.world_agents]
