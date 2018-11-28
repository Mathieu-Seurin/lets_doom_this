import random
import time
import argparse

from os import path

from config import load_config_and_ext

import gym
import gym.wrappers

import ray

import logging

episodes = 10

#game = gym.wrappers.Monitor(game, "test_out", resume=True)

@ray.remote
def test_episodes(episodes):
    import gym_vizdoom
    game = gym.make("{}-v0".format("VizdoomBasic"))

    for i in range(episodes):
        state = game.reset()
        done = False
        j = 0
        reward_total = 0
        while not done:
            action = 3 #game.action_space.sample()
            observation, reward, done, info = game.step(action=action)
            j = j+1
            reward_total += reward

            assert reward_total < 100 and reward_total > -400, "damn"

            if done:
                break

    return "lol"


num_worker = 100

ray.init()
results = ray.get([test_episodes.remote(100) for i in range(num_worker)])

print(results)
print(len(results))
