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

    def __init__(self, state_size, n_actions, batch_size):
        """
        Takes the expected size of the state-vector and the number of actions.
        """
        super(DQN, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.middle_layer_neurons = 16


        self.layer_list = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation='relu', input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.middle_layer_neurons, activation="relu"),
            tf.keras.layers.Dense(n_actions)

        ]


    # @tf.function
    def __call__(self, state):
        print("initial stat:", state)
        for layer in self.layer_list:
            state = layer(state)
            print("State:", state)

        output = {}
        output["q_values"] = state
        return output


def train(dqn, state, action, target, optim, loss_func):
    with tf.GradientTape() as tape:
        prediction = dqn(state)
        loss = loss_func(target, prediction["q_values"][action])
        gradients = tape.gradient(loss, dqn.trainable_variables)
    optim.apply_gradients(zip(gradients, dqn.trainable_variables))

    return tf.math.reduce_mean(loss)



if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    model_kwargs = {
        "batch_size": 8,
        "state_size": 4,
        "n_actions": 2
    }

    kwargs = {
        "model": DQN,
        "model_kwargs": model_kwargs,
        "environment": 'CartPole-v0',
        "num_parallel": 2,
        "total_steps": 100,
    }

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    # print("test before training: ")
    # manager.test(
    #     max_steps=100,
    #     test_episodes=10,
    #     render=True,
    #     do_print=True,
    #     evaluation_measure="time_and_reward",
    # )

    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    gamma = 0.95
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.keras.losses.MSE

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
        for key in data_dict:
            print("data:", data_dict[key])

        q_target = data_dict['reward'] + gamma * agent.max_q(data_dict['state_new'])

        # TODO: optimize agent
        loss = train(agent.model, data_dict['state'], data_dict['action'], q_target, optimizer, loss_function)


        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {loss}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        # manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)

