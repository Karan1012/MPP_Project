import random
import time
from collections import deque

from torch import optim

from a3c.model import ValueNetwork, PolicyNetwork

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import numpy as np

from utils.device import device

class A3CWorker(mp.Process):

    def __init__(self, id, env, state_size, action_size, gamma, lr, global_value_network, global_policy_network,
                 global_value_optimizer, global_policy_optimizer,
                 global_episode, n_episodes, update_every, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(A3CWorker, self).__init__()
        self.name = "w%i" % id

        self.id = id

        self.update_every = update_every

        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.env.seed(id)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = gamma
        # self.local_value_network = ValueNetwork(self.obs_dim, 1)
        # self.local_policy_network = PolicyNetwork(self.obs_dim, self.action_dim)

        self.global_value_network = global_value_network
        self.global_policy_network = global_policy_network
        self.global_episode = global_episode
     #   self.global_value_optimizer = optim.SGD(self.global_value_network.parameters(), lr=.01, momentum=.5)
      #  self.global_policy_optimizer = optim.SGD(self.global_policy_network.parameters(), lr=.01, momentum=.5)
        self.global_value_optimizer = global_value_optimizer #optim.Adam(self.global_value_network.parameters(), lr=lr)
        self.global_policy_optimizer = global_policy_optimizer #optim.Adam(self.global_policy_network.parameters(), lr=lr)
        self.n_episodes = n_episodes

        # sync local networks with global networks
       # self.sync_with_global()

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def act(self, state, eps=0.):
        if random.random() > eps:
            # Turn the state into a tensor_
         #   state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            state = torch.from_numpy(state).float().to(device)

            with torch.no_grad():
                action_values = self.global_policy_network(state)  # Make choice based on local network

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(device)


        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma ** i for i in range(rewards[j:].size(0))]) \
                                        * rewards[j:]) for j in
                              range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(device)

        # compute value loss
        values = self.global_value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())

        if np.asarray(torch.isnan(values)[0])[0] == True:
            x = 2

        # compute policy loss with entropy bonus
        logits = self.global_policy_network.forward(states)

        if np.asarray(torch.isnan(logits)[0])[0] == True:
            x= 2

        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)

        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()

        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean() - 0.001 * entropy

        return value_loss, policy_loss

    def update_global(self, trajectory):
        value_loss, policy_loss = self.compute_loss(trajectory)

        self.global_value_optimizer.zero_grad()
        value_loss.backward()
        # propagate local gradients to global parameters
        # for local_params, global_params in zip(self.local_value_network.parameters(),
        #                                        self.global_value_network.parameters()):
        #     global_params._grad = local_params._grad
        self.global_value_optimizer.step()

        self.global_policy_optimizer.zero_grad()
        policy_loss.backward()
        # # propagate local gradients to global parameters
        # for local_params, global_params in zip(self.local_policy_network.parameters(),
        #                                        self.global_policy_network.parameters()):
        #     global_params._grad = local_params._grad
            # print(global_params._grad)
        self.global_policy_optimizer.step()

    # def sync_with_global(self):
    #     self.local_value_network.load_state_dict(self.global_value_network.state_dict())
    #     self.local_policy_network.load_state_dict(self.global_policy_network.state_dict())

    def run(self):
        t_step = 0
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        start_time = time.time()
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            trajectory = []
            score = 0
            t_step += 1
            for t in range(self.max_t):
             #   self.sync_with_global()
                action = self.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append([state, action, reward, next_state, done])

              # if (len(trajectory) == 64):


                state = next_state
                score += reward
                if done:
                    with self.global_episode.get_lock():
                        self.update_global(trajectory)
                        #trajectory = []
                    # if len(trajectory):
                    #     with self.global_episode.get_lock():
                    #         self.update_global(trajectory)
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon
            elapsed_time = time.time() - start_time
            if self.id == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}, Runtime: '.format(self.id, i_episode, np.mean(
                    scores_window)) + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if i_episode % 100 == 0:
                print('\rThread: {}, Episode {}\tAverage Score: {:.2f}, Runtime: '.format(self.id, i_episode, np.mean(
                    scores_window)) + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                break

        # state = self.env.reset()
        # trajectory = []  # [[s, a, r, s', done], [], ...]
        # episode_reward = 0
        #
        # while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
        #     action = self.get_action(state)
        #     next_state, reward, done, _ = self.env.step(action)
        #     trajectory.append([state, action, reward, next_state, done])
        #     episode_reward += reward
        #
        #     if done:
        #         with self.global_episode.get_lock():
        #             self.global_episode.value += 1
        #         print(self.name + " | episode: " + str(self.global_episode.value) + " " + str(episode_reward))
        #

        #
        #         trajectory = []
        #         episode_reward = 0
        #         state = self.env.reset()
        #     else:
        #         state = next_state