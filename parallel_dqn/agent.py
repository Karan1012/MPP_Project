from multiprocessing import Queue, Pipe

from utils.agent import Agent
from parallel_dqn.parameter_server import ParameterServer
from parallel_dqn.worker import ParallelDQNWorker


class ParallelDQNAgent(Agent):
    def __init__(self, env, num_threads, gamma, lr, global_max_episode):


        grads_q = Queue()
        current_q = Queue()

        send_conn, recv_conn = Pipe()

        ps = ParameterServer(grads_q, current_q, recv_conn, env.observation_space.shape[0], env.action_space.n, 0, num_threads)
        ps.start()

        self.workers = [ParallelDQNWorker(id=id, env=env, send_conn=send_conn, state_size=env.observation_space.shape[0],
                      action_size=env.action_space.n, grads_q=grads_q, current_q=current_q, n_episodes=global_max_episode, lr=lr, gamma=gamma) for id in range(num_threads)]


    def train(self):
        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]

    # def save_model(self):
    #     torch.save(self.global_value_network.state_dict(), "a3c_value_model.pth")
    #     torch.save(self.global_policy_network.state_dict(), "a3c_policy_model.pth")
