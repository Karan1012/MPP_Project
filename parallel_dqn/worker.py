import random
from collections import deque
from multiprocessing import Process

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.pyplot import plot

from parallel_dqn.model import QNetwork
from parallel_dqn.replay_buffer import ReplayBuffer
import torch.multiprocessing as mp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size

TAU = 1e-3  # for soft update of target parameters

UPDATE_EVERY = 5  # how often to update the network

class ParallelDQNWorker(Process):

    def __init__(self, id, env, send_conn, state_size, action_size, grads_q, current_q, n_episodes, lr, gamma, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(ParallelDQNWorker, self).__init__()
        self.id = id
        self.env = env
        self.grads_q = grads_q
        self.state_size = state_size
        self.action_size = action_size
        self.send_conn = send_conn
        self.current_q = current_q
        self.n_episodes = n_episodes
        self.gamma = gamma

        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, 0)

        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=lr)

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Turn the state into a tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Takes the network out of training mode in order to generate action based on state
            #   self.qnetwork.eval()

            with torch.no_grad():
                action_values = self.qnetwork_local(state)  # Make choice based on local network

            #      self.qnetwork.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        #   self.conn.send((state, action, reward, next_state, done))
        self.memory.add(state, action, reward, next_state, done)

        # lock
        try:
            params = self.current_q.get(False)
            self.current_q.put(params)


            for local, server in zip(self.qnetwork_local.parameters(), params):
                local.data.copy_(server.data)

            if self.t_step % UPDATE_EVERY == 0:
                for local, server in zip(self.qnetwork_target.parameters(), params):
                    local.data.copy_(server.data)
        except:
            pass

        #    for local, server in zip(self.qnetwork_target.parameters(), ParameterServer.get_parameters()):
        #       local.data.copy_(server.data)
        # self.qnetwork_local = ParameterServer.model

        # self.soft_update(self.qnetwork_local.parameters(), ParameterServer.get_parameters(), TAU)
        # self.qnetwork_local.set_weights(ParameterServer.get_weights())

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(device)
            self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # existing_shm = shared_memory.SharedMemory(name='psm_21467_46075')
        # c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)

        # ------------------- update target network ------------------- #
        # self.qs[self.id].put((self.id, self.qnetwork_local.get_gradients()))
        # ParameterServer.apply_gradients()

        # ParameterServer.soft_update(self.qnetwork_local, TAU)

        #  if ParameterServer.get_n:
        #  self.soft_update(self.qnetwork_target.parameters(),  ParameterServer.get_parameters(), TAU)
        # self.qnetwork_target = ParameterServer.model

        # print(self.qnetwork_local.get_gradients())
        buf = self.qnetwork_local.get_gradients()
        self.grads_q.put((self.id, buf))



        # while not self.update_q.empty():
        #     try:
        #         params = self.current_q.get(False)
        #         #self.soft_update(params, self.qnetwork_target.parameters(), TAU)
        #         for local, server in zip(self.qnetwork_target.parameters(), params):
        #             local.data.copy_(server.data)
        #     except:
        #         pass

        # ------------------- update target network ------------------- #



    #  self.send_conn.close()

    # if self.t_step % UPDATE_EVERY == 0:
    # ParameterServer.apply_gradients()

    # ParameterServer.soft_update(TAU)
    # self.soft_update(ParameterServer.get_parameters(), self.qnetwork_target.parameters(), TAU)
    #  existing_shm = shared_memory.SharedMemory(name='parameters')
    #  c = np.ndarray(ParameterServer.shmsize, dtype=ParameterServer.shmtype, buffer=existing_shm.buf)
    #     for local, server in zip(self.qnetwork_target.parameters(), c):
    #        local.data.copy_(server.data)

    # self.Q[next_state][action] = (1 - self.learning_rate) * self.Q[state][action] + self.learning_rate * (reward + self.discounting_factor * np.argmax(self.Q[next_state]))
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model, local_model):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # def run(self):
    #     state = self.env.reset()
    #     trajectory = []  # [[s, a, r, s', done], [], ...]
    #     episode_reward = 0
    #
    #     while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
    #         action = self.get_action(state)
    #         next_state, reward, done, _ = self.env.step(action)
    #         trajectory.append([state, action, reward, next_state, done])
    #         episode_reward += reward
    #
    #         if done:
    #             with self.global_episode.get_lock():
    #                 self.global_episode.value += 1
    #             print(self.name + " | episode: " + str(self.global_episode.value) + " " + str(episode_reward))
    #
    #             self.update_global(trajectory)
    #             self.sync_with_global()
    #
    #             trajectory = []
    #             episode_reward = 0
    #             state = self.env.reset()
    #         else:
    #             state = next_state

    def run(self):
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.max_t):
                action = self.act(state, eps)
                # if do_render:
                #     self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            if self.id == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}'.format(self.id, i_episode, np.mean(scores_window)))
            if i_episode % 100 == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}'.format(self.id, i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                torch.save(self.qnetwork_local.state_dict(), 'checkpoint.pth')
                break

        plot(id, scores)