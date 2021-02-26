import os 
import gym
import numpy as np
import tensorflow as tf
import ray
from really import SampleManager
from gridworlds import GridWorld
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


"""
DQN homework
"""


class DQN(tf.keras.Model):

    def __init__(self, state_size, n_actions):
        """
        Takes the expected size of the state-vector and the number of actions.
        """
        super(DQN, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.middle_layer_neurons = 32

        self.layer_list = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation='relu', input_shape=(None, state_size)),
            tf.keras.layers.Dense(self.middle_layer_neurons, activation="relu"),
            tf.keras.layers.Dense(n_actions)

        ]


    @tf.function
    def __call__(self, state):
        for layer in self.layer_list:
            state = layer(state)

        output = {}
        output["q_values"] = state
        return output




if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    kwargs = {
        "model": DQN,
        "environment": 'CartPole-v0',
        "num_parallel": 4,
        "total_steps": 100,
    }

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )

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
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, do_print=True, render=True)

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
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        # TODO: iterate through your datasets

        # TODO: optimize agent

        dummy_losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]

        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)

