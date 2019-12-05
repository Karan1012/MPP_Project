import random
import time
from collections import deque
from multiprocessing import Process
TAU = 1e-3  # for soft update of target parameters
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from parallel_dqn.model import QNetwork
from parallel_dqn.replay_buffer import ReplayBuffer
import torch.multiprocessing as mp

from parallel_dqn.utils import copy_parameters
from plot import plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
MAX_LOCAL_MEMORY = 10


class ParallelDQNWorker(mp.Process):

    def __init__(self, id, env, ps, state_size, action_size, n_episodes, lr, gamma, update_every, global_network, target_network, optimizer, lock, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(ParallelDQNWorker, self).__init__()
        self.id = id
        self.ps = ps
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.update_every = update_every
       # self.global_memory = global_memory
       # self.local_memory = mp.Queue()
        self.local_memory = ReplayBuffer(env.action_space.n, BUFFER_SIZE, BATCH_SIZE)

        self.global_network = global_network
        self.qnetwork_target = target_network
  #      self.global_optimizer = global_optimizer


       # self.qnetwork_local = QNetwork(state_size, action_size).to(device)


        #self.ps.initialize_gradients(self.id, [p for p in self.qnetwork_target.parameters()])

       # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.optimizer = optimizer

        self.l = lock





    def act(self, state, eps=0.):

        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Turn the state into a tensor_
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action_values = self.global_network(state)  # Make choice based on local network

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.local_memory.add(state, action, reward, next_state, done)

        # Update local parameters with that of parameter server
        #copy_parameters(self.ps.get_parameters(), self.qnetwork_local.parameters())
        #summed_gradients = torch.tensor(self.ps.get_summed_gradients())

        #self.qnetwork_local.set_gradients(self.ps.sync())
     #   copy_parameters(self.ps.get(), self.qnetwork_local.parameters())
     #    self.l.acquire()
     #    try:
     #        self.qnetwork_local.load_state_dict(self.global_network.state_dict())
     #    finally:
     #        self.l.release()



        # Increment local timer
        self.t_step += 1

        # if self.t_step % MAX_LOCAL_MEMORY == 0:
        #     self.global_memory.add(self.local_memory)

        # If enough samples are available in memory, get random subset and learn
        # Learn every UPDATE_EVERY time steps.
        if self.t_step % self.update_every == 0:
            if self.t_step > BATCH_SIZE:
                experiences = self.local_memory.sample(BATCH_SIZE)
                self.learn(experiences)

        # Learn every UPDATE_EVERY time steps.
    #    if self.t_step % self.update_every == 0: # TODO: Fix, need to make global memory first
            #summed_gradients = torch.tensor(self.ps.get_summed_gradients()) # copy shared memory tensor back to local memory
            #self.qnetwork_target.set_gradients(self.ps.sync())
            #copy_parameters(self.ps.get(), self.qnetwork_target.parameters())
      #      self.qnetwork_target.load_state_dict(self.global_network.state_dict())
            # self.l.acquire()
            # try:
            #     self.qnetwork_target.load_state_dict(self.global_network.state_dict())
            # finally:
            #     self.l.release()



    def learn(self, experiences):
        #self.l
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        self.l.acquire()
        try:


            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
           # Q_expected = self.qnetwork_local(states).gather(1, actions)
            Q_expected = self.global_network(states).gather(1, actions)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.global_network, self.qnetwork_target, TAU)

        finally:
            self.l.release()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def run(self):
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        start_time = time.time()
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.max_t):
                action = self.act(state, eps)
               # iclearn
                # f do_render:
               #      self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            elapsed_time = time.time() - start_time
            if self.id == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}, Runtime: '.format(self.id, i_episode, np.mean(scores_window)) + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if i_episode % 100 == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}, Runtime: '.format(self.id, i_episode, np.mean(scores_window)) + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                torch.save(self.global_network.state_dict(), 'checkpoint.pth')
                break

       # plot(id, scores)
