import random
from collections import deque, namedtuple
from multiprocessing import Process

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.pyplot import plot

from dynaq.env_model import EnvModelNetwork
from dynaq.priority import PriorityModel
from parallel_dqn.model import QNetwork
from parallel_dqn.replay_buffer import ReplayBuffer
import torch.multiprocessing as mp

from parallel_dqn.utils import copy_parameters
from utils.avg import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TAU = 1e-3

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size


THETA = 0.0001

class DynaQWorldWorker(mp.Process):

    def __init__(self, id, env, state_size, action_size, n_episodes, lr, gamma, update_every,global_network, target_network, world_model, q, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(DynaQWorldWorker, self).__init__()
        self.id = id
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.update_every = update_every
        self.q = q

       # self.env_model = env_model

        # TODO: It would probably be better if this was shared between all processes
        # However, that might hurt performance
        # Could instead maybe create one global memory and send a few experiences in batches?
        self.local_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
        self.simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
        self.initial_states = 0

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.env_model = world_model #EnvModelNetwork(state_size, action_size).to(device)

        self.planning_steps = 10
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.global_network = global_network
        self.target_network = target_network
        self.world_model = world_model

        self.world_optimizer = optim.SGD(self.world_model.parameters(), lr=lr, momentum=.5)

        self.optimizer = optim.SGD(self.global_network.parameters(), lr=lr, momentum=.5)

        self.initial_states = []

        self.losses = AverageMeter()


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
        # if t == 0:
        #     self.initial_states.append(state)

        # Save experience in replay memory
        self.local_memory.add(state, action, reward, next_state, done)

        # Update local parameters with that of parameter server
        #copy_parameters(self.ps.get_parameters(), self.qnetwork_local.parameters())
        #summed_gradients = torch.tensor(self.ps.get_summed_gradients())

        #self.qnetwork_local.set_gradients(self.ps.sync())
        copy_parameters(self.ps.get(), self.global_network.parameters())

        # Increment local timer
        self.t_step += 1

        # if self.t_step % MAX_LOCAL_MEMORY == 0:
        #     self.global_memory.add(self.local_memory)

        # If enough samples are available in memory, get random subset and learn
        # Learn every UPDATE_EVERY time steps.
        #if self.t_step % self.update_every == 0:
        if self.t_step > BATCH_SIZE:
            experiences = self.local_memory.sample(BATCH_SIZE)
            self.learn(experiences)
            #self.learn_world(experiences)

            experiences = self.local_memory.sample(BATCH_SIZE)
            self.learn_world(experiences)

        if self.t_step % 1000 == 0:
            print("predicting on world")
            self.planning()


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
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




    def learn_world(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        act = self.world_model.encode_action(actions)

        self.world_optimizer.zero_grad()

        out1, out2, out3 = self.world_model(states, act)

        loss1 = F.mse_loss(out1, next_states)
        loss2 = F.mse_loss(out2, rewards)
       # loss3 = F.mse_loss(out3, dones)
        loss3 = F.binary_cross_entropy(out3, dones)
        loss = loss1 + loss2 + loss3


        loss.backward()

        self.world_optimizer.step()

        self.losses.update(np.array(loss.data).reshape(1)[0])

       # print("World model loss: ", loss)


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

    def planning(self):

        simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)

        for i in range(1):
            done = False
            state = self.local_memory.get_one_state() #self.initial_states[random.choice(np.arange(len(self.initial_states)))]
            num_steps = 0

            while not done:
                num_steps += 1
                action = random.choice(np.arange(self.action_size))

                s_ = torch.from_numpy(np.vstack([state])).to(device)
                a_ = self.world_model.encode_action(torch.from_numpy(np.vstack([[action]])).to(device))

                with torch.no_grad():
                    next_state, reward, done_ = self.world_model(s_, a_)

                #done = F.softmax(done_)[0][0]
                done = True if np.asarray(done_)[0][0] >= .5 else False

                simulated_memory.add(state, action, reward[0][0], next_state[0], done)

                state = next_state

                if num_steps > 100:
                    done = True

        l = min(BATCH_SIZE, len(simulated_memory))
        experiences = simulated_memory.sample(l)
        self.learn(experiences)

    def run(self):
        t_step = 0
        while True:
            t_step += 1
            if t_step % 100 == 0:
                print("World loss: ", self.losses.avg)
                self.planning()
                self.losses.reset()



            experiences = self.q.get()

            for (state, action, next_state, reward, done) in  list(zip(*experiences)):
                self.local_memory.add(state, action, next_state, reward, done)

            self.learn_world(experiences)

          #  if t_step > 1000:
                #print("planning step")







        #plot(id, scores)