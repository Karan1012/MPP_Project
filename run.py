import multiprocessing
import threading
from multiprocessing import Process, Queue, Array

import gym
import torch
import numpy as np
from collections import deque
import argparse

from random import randint, seed

#from dqn.agent import DQNAgent as Agent
from parallel_dqn.parameter_server import ParameterServer
from plot import plot
#from q.agent import QAgent as Agent
from parallel_dqn.agent import ParallelDQNAgent as Agent

seed(1)


""" Train"

Params
======
    n_episodes (int): maximum number of training episodes
    max_t (int): maximum number of timesteps per episode
    eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
"""
def train(id, qs, do_render, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    env = gym.make('LunarLander-v2')
    env.seed(randint(0,100))

    agent = Agent(id=id, state_size=env.observation_space.shape[0], action_size=env.action_space.n, qs=qs, seed=randint(0, 100))

    # print('State shape: ', env.observation_space.shape)
    # print('Number of actions: ', env.action_space.n)

     # list containing scores from each episode
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            if do_render:
                env.render()
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rThread: {}, Episode {}\tAverage Score: {:.2f}'.format(id, i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rThread: {}, Episode {}\tAverage Score: {:.2f}'.format(id, i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break


    #while not qs[id].empty():
    #     qs[id].get()
    #
    # qs[id].close()
  #  qs[id].cancel_join_thread()

    env.close()

    plot(id, scores)

    qs[id].cancel_join_thread()


def main():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--do-render', type=bool, default=False, help='Whether or not to render game')
    args = parser.parse_args()

    scores = [[] for _ in range(args.num_threads)]
   # scores = [Array('d', range(args.num_threads)) for

    env = gym.make('LunarLander-v2')

    # manager = multiprocessing.Manager()
    # l = manager.list(range(args.num_threads))
    qs = [Queue() for _ in range(args.num_threads)]

    ParameterServer(env.observation_space.shape[0], env.action_space.n, 0, qs, args.num_threads)
    processes = [Process(target=train, args=(i, qs, args.do_render, args.num_episodes)) for i in range(args.num_threads)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # if args.num_threads == 1:
    #     scores = [scores]
    #
    # plot(scores)
    env.close()


if __name__ == "__main__":
    main()





