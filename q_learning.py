import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import gym
import ray
from q_model import QTable

from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    if not os.path.exists("./progress_test/"):
        os.makedirs("progress_test")

    kwargs = {
        "model": QTable,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets

        print(sample_dict)

        print("optimizing...")

        # TODO: iterate through your datasets

        # TODO: optimize agent


        # set new weights
        # manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        #manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load models
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
