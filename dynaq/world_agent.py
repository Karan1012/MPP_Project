import random
from collections import namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

from dqn.replay_buffer import ReplayBuffer
from utils.avg import Average

TAU = 1e-3

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size


THETA = 0.0001

class DynaQWorldAgent(mp.Process):

    def __init__(self, id, state_size, action_size, n_episodes, lr, gamma, global_network, target_network, world_model, world_optimizer, world_lock, q, lock, num_threads):
        super(DynaQWorldAgent, self).__init__()
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.n_episodes = n_episodes
        self.gamma = gamma

        self.q = q
        self.l = lock
        self.num_threads = num_threads
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.local_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
        self.simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)


        self.world_model = world_model

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.global_network = global_network
        self.target_network = target_network
        self.world_model = world_model

        self.world_optimizer = world_optimizer #optim.SGD(self.world_model.parameters(), lr=lr, momentum=.5)
        self.world_lock = world_lock

        self.optimizer = optim.SGD(self.global_network.parameters(), lr=lr, momentum=.5)

        self.losses = Average()


    def act(self, state, eps=0.):
        if random.random() > eps:
            # Turn the state into a tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_values = self.global_network(state)  # Make choice based on local network

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


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

    def act(self, state, eps=.1):
        if random.random() > eps:
            # Turn the state into a tensor
          #  state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_values = self.global_network(state)  # Make choice based on local network

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn_world(self, states, actions, rewards, next_states, dones):
        act = self.world_model.encode_action(actions)

        self.world_optimizer.zero_grad()

        out1, out2, out3 = self.world_model(states, act)

        loss1 = F.mse_loss(out1, next_states)
        loss2 = F.mse_loss(out2, rewards)
        loss3 = F.binary_cross_entropy(out3, dones)
        loss = loss1 + loss2 + loss3
        loss.backward()

        self.world_optimizer.step()

        self.losses.update(np.array(loss.data).reshape(1)[0])

    def get_experience_as_tensor(self, e):
        states = torch.from_numpy(np.vstack([e.state])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def get_action_values(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_values = self.target_network(state)

        return action_values.cpu().data.numpy()[0]

    def get_delta(self, state, action, next_state, reward):
        priority = reward + self.gamma * np.max(self.get_action_values(next_state)) - self.get_action_values(state)[action]
        return priority

    def planning(self):

        simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)

        for i in range(1):
            done = False
            state = self.local_memory.get_one_state()
            num_steps = 0

            while not done:
                num_steps += 1
                action = self.act(state) #random.choice(np.arange(self.action_size))

                s_ = torch.from_numpy(np.vstack([state])).to(self.device)
                a_ = self.world_model.encode_action(torch.from_numpy(np.vstack([[action]])).to(self.device))

                with torch.no_grad():
                    next_state, reward, done_ = self.world_model(s_, a_)

                done = True if np.asarray(done_)[0][0] >= .5 else False

                simulated_memory.add(state, action, reward[0][0], next_state[0], done)

                state = next_state

                if num_steps > 100:
                    done = True

        l = min(BATCH_SIZE, len(simulated_memory))
        experiences = simulated_memory.sample(l)
        self.learn(experiences)

    # TODO: Make this better
    def get_from_queue(self):
        self.l.acquire()
        try:
            states_ = self.q[0].get()
            actions_ = self.q[1].get()
            rewards_ = self.q[2].get()
            next_states_ = self.q[3].get()
            dones_ = self.q[4].get()

            states = states_.clone()
            actions = actions_.clone()
            rewards = rewards_.clone()
            next_states = next_states_.clone()
            dones = dones_.clone()

            del states_
            del actions_
            del rewards_
            del next_states_
            del dones_

        finally:
            self.l.release()

        return states, actions, rewards, next_states, dones

    def run(self):
        t_step = 0
        while True:

            t_step += 1

            states, actions, rewards, next_states, dones = self.get_from_queue()

            if t_step % 100 == 0:
                print("World loss: ", self.losses.avg)
                self.losses.reset()

            for (state, action, next_state, reward, done) in zip(states, actions, rewards, next_states, dones):
                self.local_memory.add(state, action, next_state, reward, done)

            self.world_lock.acquire()
            try:
                self.learn_world(states, actions, rewards, next_states, dones)
            finally:
                self.world_lock.release()

            if self.losses.avg < 50 and (t_step % 100 == 0):
                self.planning()

