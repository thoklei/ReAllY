import logging, os

import gym
import numpy as np
import tensorflow as tf
import ray
from really import SampleManager
# from really.utils import (
#     dict_to_dict_of_datasets,
# )

from gridworlds.envs.gridworld import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):

    def __init__(self, h, w, action_space):
        self.action_space = action_space
        self.table = np.zeros((action_space, h, w))
        

    def __call__(self, state):
        state = np.squeeze(state)
        x,y = state
        x = int(x)
        y = int(y)

        output = {}
        output["q_values"] = np.expand_dims(self.table[:, x,y], axis=0)
        # TODO return v_estimate

        return output
        

    def get_weights(self):
        return self.table.copy()

    def set_weights(self, q_vals):
        # print("Q-vals in set_weights: ", q_vals)
        self.table = q_vals.copy()



if __name__ == "__main__":

    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 200,
        "model_kwargs": model_kwargs,
        "action_sampling_type": "epsilon_greedy",
        "epsilon": 0.7,
        "env_kwargs": env_kwargs
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    # do the rest!!!!
    epochs = 30
    buffer_size = 5000
    test_steps = 1000
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    alpha = 0.1
    gamma = 0.95

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]
    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)
    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    print("test before training: ")
    manager.test(
        max_steps=10,
        test_episodes=1,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        # data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        # iterate through your datasets
        old_table = agent.get_weights()
        for s, a , r , n , d in zip(sample_dict['state'], sample_dict['action'], sample_dict['reward'], sample_dict['state_new'], sample_dict['not_done']):
            print(s, a , r , n , d )
            s_x, s_y = s
            n_x, n_y = n
            # update weights
            old_table[a, s_x, s_y] += alpha * (r + gamma * np.max(old_table[:, n_x, n_y]) - old_table[a, s_x, s_y])

        # optimize agent
        # calculate losses
        weights = agent.model.get_weights()
        losses = (old_table-weights)**2

        # set new weights
        manager.set_agent(old_table)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        # manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            # manager.save_model(saving_path, e)
            pass

    # and load models
    # manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=3, render=True)