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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size


THETA = 0.0001

class DynaQWorker(mp.Process):

    def __init__(self, id, env, ps, state_size, action_size, n_episodes, lr, gamma, update_every, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        super(DynaQWorker, self).__init__()
        self.id = id
        self.ps = ps
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.update_every = update_every

       # self.env_model = env_model

        # TODO: It would probably be better if this was shared between all processes
        # However, that might hurt performance
        # Could instead maybe create one global memory and send a few experiences in batches?
        self.local_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
        self.simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
        self.initial_states = 0
        #
        # self.simulated_env = gym.make('LunarLander-v2')
        # self.simulated_env.reset()


        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.model = PriorityModel()

        self.t_step = 0
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.env_model = EnvModelNetwork(state_size, action_size).to(device)

        self.planning_steps = 10
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.world_model = EnvModelNetwork(state_size, action_size).to(device)
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=lr)

        self.initial_states = []


    def act(self, state, eps=0.):
        if random.random() > eps:
            # Turn the state into a tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action_values = self.qnetwork_local(state)  # Make choice based on local network

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
        copy_parameters(self.ps.get(), self.qnetwork_local.parameters())

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

            experiences = self.local_memory.sample(BATCH_SIZE)
            self.learn_world(experiences)

        if self.t_step % 1000 == 0:
            print("predicting on world")
            self.planning()

        # Learn every UPDATE_EVERY time steps.
        if self.t_step % self.update_every == 0: # TODO: Fix, need to make global memory first
            #summed_gradients = torch.tensor(self.ps.get_summed_gradients()) # copy shared memory tensor back to local memory
            #self.qnetwork_target.set_gradients(self.ps.sync())
            copy_parameters(self.ps.get(), self.qnetwork_target.parameters())

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

        self.ps.record_gradients(self.qnetwork_local.get_gradients())


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
            action_values = self.qnetwork_target(state)

        return action_values.cpu().data.numpy()[0]

    def get_delta(self, state, action, next_state, reward):
        priority = reward + self.gamma * np.max(self.get_action_values(next_state)) - self.get_action_values(state)[action]
        return priority

    def planning(self):

        simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)

        for i in range(5):
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


     #   # simulated_memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
     #    #state = self.simulated_env.reset()
     #
     #   # for i in range(BATCH_SIZE):
     #    states, actions, rewards, next_states, dones = self.local_memory.sample(BATCH_SIZE)
     #
     #    a = [[random.choice(np.arange(self.action_size))] for _ in range(BATCH_SIZE)]
     #
     #    a_ = torch.from_numpy(np.vstack(a)).long().to(device)
     #        # while a == action:
     #        #     a = random.choice(np.arange(self.action_size))
     #
     #    act = self.world_model.encode_action(a_)
     #
     #    with torch.no_grad():
     #        next_state, reward, done = self.world_model(states, act)
     #
     #        #next_state, reward, done, _ = self.world_model.
     #        #experience = self.local_memory.get_one_experience()
     #
     #       # states, actions, rewards, next_states, dones = experiences
     #        #state = self.initial_states[random.choice(np.arange(len(self.initial_states)))]
     #        # action = random.choice(np.arange(self.action_size))
     #        #
     #        # action_values = self.get_action_values(experience.state)
     #        # reward = action_values[action]
     #
     #        #simulated_memory.add(states, a, reward, next_state, done)
     #
     #        # if done:
     #        #     state = self.simulated_env.reset()
     #        # else:
     #        #     state = next_state
     #
     # #   experiences = simulated_memory.sample(BATCH_SIZE)
     #    experiences = (states, actions, rewards, next_states, dones)
     #    self.learn(experiences)

    def planning1(self, state, action, next_state, reward, done):
        # feed the model with experience
        self.model.feed(state, action, next_state, reward)

        # get the priority for current state action pair
        priority = np.abs(self.get_delta(state, action, next_state, reward))

        if priority > THETA:
            self.model.insert(priority, state, action, next_state, reward)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < self.planning_steps and not self.model.empty():
            # get a sample with highest priority from the model
            priority, state_, action_, next_state_, reward_ = self.model.sample()

            # update the state action value for the sample
           # delta = self.get_delta(state_, action_, next_state_, reward_)

            #q_value[state_[0], state_[1], action_] += self.alpha * delta
            e = self.experience(state_, action_, reward_, next_state, done)

            self.learn(self.get_experience_as_tensor(e))

            # deal with all the predecessors of the sample state
            # for state_pre, action_pre, reward_pre in self.model.predecessor(state_):
            #     priority = np.abs(self.get_delta(state_pre, action_pre, state_, reward_pre))
            #
            #     if priority > THETA:
            #         self.model.insert(priority, state_pre, action_pre)

            planning_step += 1

        # for _ in range(N):
        #     experiences = self.local_memory.sample(1)
        #     self.learn(experiences)
       # states, actions, rewards, next_states, dones = experiences

       # rewards, next_states =


    def run(self):
        scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start  # initialize epsilon
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            self.initial_states.append(state)
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