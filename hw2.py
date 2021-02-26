import gym
import numpy as np
import tensorflow as tf
import ray
from really import SampleManager
from gridworlds import GridWorld

"""
DQN homework
"""


class DQN(tf.keras.model):

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


    # @tf.function
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

    # do the rest!!!!
