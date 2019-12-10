from dqn.agent import DQNAgent
from dqn.model import QNetwork
from utils.rl_algorithm import RLAlgorithm

UPDATE_EVERY = 5
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size

class DQN(RLAlgorithm):
    def __init__(self, env, do_render, num_threads, gamma, lr, global_max_episode):

        state_size, action_size = env.observation_space.shape[0], env.action_space.n

        self.qnetwork_global = QNetwork(state_size, action_size) #.to(device)
        self.qnetwork_global.share_memory()

        self.qnetwork_target = QNetwork(state_size, action_size) #.to(device)
        self.qnetwork_target.share_memory()

        self.agents = [DQNAgent(id=id, env=env, do_render=do_render, state_size=state_size,
                      action_size=action_size, n_episodes=global_max_episode, lr=lr, gamma=gamma, update_every=UPDATE_EVERY + num_threads,
                                          global_network=self.qnetwork_global, target_network=self.qnetwork_target) for id in range(num_threads)]

    def train(self):
        [agent.start() for agent in self.agents]
        [agent.join() for agent in self.agents]

