import numpy as np
from really import SampleManager
import os
import gridworlds
import gym
import random

env = gym.make('gridworld-v0')
env.reset()
a = [1,1,1,1,2,2,2,2,2]
done = False
while not(done):
    #print(i)
    action = random.randint(0,3)
    #rint(action)
    state, reward, done, info = env.step(action)
    env.render()
    print('reward', reward)
    if done:
        print('done')
        break
#for i in range(10):
#    state, reward, done, info = env.step(i)
#    env.render()
#    print('reward', reward)
