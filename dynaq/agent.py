import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

from dqn.replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TAU = 1e-3

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size


THETA = 0.0001

class DynaQAgent(mp.Process):

    def __init__(self, id, env, state_size, action_size, n_episodes, lr, gamma, global_network, target_network, q, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(DynaQAgent, self).__init__()
        self.id = id
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.n_episodes = n_episodes
        self.gamma = gamma

        self.q = q

        self.local_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.global_network = global_network
        self.target_network = target_network

        self.optimizer = optim.SGD(self.global_network.parameters(), lr=lr, momentum=.5)

        self.scores_window = deque(maxlen=100)  # last 100 scores


    def act(self, state, eps=0.):
        if random.random() > eps:
            # Turn the state into a tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action_values = self.global_network(state)  # Make choice based on local network

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.local_memory.add(state, action, reward, next_state, done)

        # Increment local timer
        self.t_step += 1


        if self.t_step > BATCH_SIZE:
            experiences = self.local_memory.sample(BATCH_SIZE)
            self.learn(experiences)

            # TODO: Better way to do this??
            if self.q[0].empty() and np.mean(self.scores_window) < 180:
                experiences = self.local_memory.sample(BATCH_SIZE)
                self.q[0].put(experiences[0].detach().share_memory_())
                self.q[1].put(experiences[1].detach().share_memory_())
                self.q[2].put(experiences[2].detach().share_memory_())
                self.q[3].put(experiences[3].detach().share_memory_())
                self.q[4].put(experiences[4].detach().share_memory_())



    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.global_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.global_network, self.target_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def get_experience_as_tensor(self, e):
        states = torch.from_numpy(np.vstack([e.state])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def get_action_values(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_values = self.target_network(state)

        return action_values.cpu().data.numpy()[0]

    def get_delta(self, state, action, next_state, reward):
        priority = reward + self.gamma * np.max(self.get_action_values(next_state)) - self.get_action_values(state)[action]
        return priority


    def run(self):
        scores = []

        eps = self.eps_start  # initialize epsilon
        start_time = time.time()
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
            self.scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            elapsed_time = time.time() - start_time
            if self.id == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}, Runtime: '.format(self.id, i_episode, np.mean(
                    self.scores_window)) + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if i_episode % 100 == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}, Runtime: '.format(self.id, i_episode, np.mean(
                    self.scores_window)) + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if np.mean(self.scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(self.scores_window)))
                break
