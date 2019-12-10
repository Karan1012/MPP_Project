from multiprocessing import Lock

from torch import optim

from parallel_dqn.model import QNetwork
from parallel_dqn.parameter_server import ParameterServer
from parallel_dqn.replay_buffer import ReplayBuffer
from parallel_dqn.worker import ParallelDQNWorker
from utils import device
from utils.agent import Agent
from utils.shared_adam import SharedAdam
from utils.shared_asgd import SharedASGD

UPDATE_EVERY = 5
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size

class ParallelDQNAgent(Agent):
    def __init__(self, env, do_render, num_threads, gamma, lr, global_max_episode):

        state_size, action_size = env.observation_space.shape[0], env.action_space.n

        self.qnetwork_global = QNetwork(state_size, action_size) #.to(device)
        self.qnetwork_global.share_memory()

        self.qnetwork_target = QNetwork(state_size, action_size) #.to(device)
        self.qnetwork_target.share_memory()

        self.workers = [ParallelDQNWorker(id=id, env=env, do_render=do_render, state_size=state_size,
                      action_size=action_size, n_episodes=global_max_episode, lr=lr, gamma=gamma, update_every=UPDATE_EVERY + num_threads,
                                          global_network=self.qnetwork_global, target_network=self.qnetwork_target) for id in range(num_threads)]

    def train(self):
       # self.ps.start()

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]

