import random
import time
from collections import deque

TAU = 1e-3  # for soft update of target parameters
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn.replay_buffer import ReplayBuffer
import torch.multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
MAX_LOCAL_MEMORY = 10


class DQNAgent(mp.Process):

    def __init__(self, id, env, do_render, state_size, action_size, n_episodes, lr, gamma, update_every, global_network, target_network, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(DQNAgent, self).__init__()
        self.id = id
        self.env = env
        self.do_render = do_render
        self.state_size = state_size
        self.action_size = action_size
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.update_every = update_every

        self.local_memory = ReplayBuffer(env.action_space.n, BUFFER_SIZE, BATCH_SIZE)

        self.global_network = global_network
        self.qnetwork_target = target_network

        self.optimizer = optim.SGD(self.global_network.parameters(), lr=lr, momentum=.5)

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay


    def act(self, state, eps=0.):
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action_values = self.global_network(state)

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.local_memory.add(state, action, reward, next_state, done)

        # Increment local timer
        self.t_step += 1

        # If enough samples are available in memory, get random subset and learn
        # Learn every UPDATE_EVERY time steps.
        if self.t_step % self.update_every == 0:
            if self.t_step > BATCH_SIZE:
                experiences = self.local_memory.sample(BATCH_SIZE)
                self.learn(experiences)

    def compute_loss(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target.forward(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_expected = self.global_network.forward(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss


    def learn(self, experiences):

        loss = self.compute_loss(experiences)

        # Update gradients per HogWild! algorithm
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.global_network, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
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
                if self.do_render:
                     self.env.render()
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

