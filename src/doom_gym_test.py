import random
import time
import argparse

from os import path

from config import load_config_and_ext

import gym
import gym_vizdoom
import gym.wrappers


import logging

game = gym.make("{}-v0".format("VizdoomBasic"))
episodes = 20000

#game = gym.wrappers.Monitor(game, "test_out", resume=True)

for i in range(episodes):

    state = game.reset()
    done = False
    j = 0
    reward_total = 0
    while not done:
        action = random.randint(0,1) #game.action_space.sample()
        observation, reward, done, info = game.step(action=action)
        j = j+1
        reward_total += reward
        print(reward_total)
        if done:
            break