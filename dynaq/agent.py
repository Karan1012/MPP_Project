from utils.agent import Agent


class DynaQAgent(Agent):
    def __init__(self, env, num_threads, gamma, lr, global_max_episode):


        self.workers = [ParallelDQNWorker(id=id, env=env, ps=self.ps, state_size=env.observation_space.shape[0],
                      action_size=env.action_space.n, n_episodes=global_max_episode, lr=lr, gamma=gamma, update_every=UPDATE_EVERY) for id in range(num_threads)]

    def train(self):
        self.ps.start()

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]