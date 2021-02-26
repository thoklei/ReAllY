import gym
import numpy as np
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
        
        super(MyModel, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.midle_layer_neurons = 32

        self.layer1 = tf.keras.layers.Dense(state_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(n_actions)


    def __call__(self, state):

        l1 = self.layer(state)
        l2 = self.layer2(l1)

        output = {}
        output["q_values"] = l2
        return output


    def get_weights(self):
        return None

    def set_weights(self, q_vals):
        pass



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
