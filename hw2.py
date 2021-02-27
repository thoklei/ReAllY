import os
import logging
import gym
import numpy as np
import tensorflow as tf
import ray
from really import SampleManager
from gridworlds import GridWorld
from really.utils import (
    dict_to_dataset,
) 

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DQN(tf.keras.Model):

    def __init__(self, state_size, n_actions, batch_size):
        """
        Constructs a Deep Q-Network.

        state_size: dimension of the state-vector
        n_actions: number of actions that can be taken
        batch_size: size of batches during training
        """

        super(DQN, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.middle_layer_neurons = 16

        self.layer_list = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation='relu', input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.middle_layer_neurons, activation="relu"),
            tf.keras.layers.Dense(n_actions)]


    # @tf.function
    def __call__(self, state):
        """
        Calculates the Q-values for all actions for a given state.

        state: the state vector
        """
        for layer in self.layer_list:
            state = layer(state)

        output = {}
        output["q_values"] = state
        return output


def train(dqn, state, action, target, optim, loss_func):
    """
    Trains the deep Q-Network.

    dqn: the network
    state: the state in which an action was taken
    action: the selected action
    target: the q-values according to the Watkins-Formula
    optim: the optimizer to be used (e.g. Adam)
    loss_func: the loss function to be used (MSE)
    """
    
    # converting actions into usable indices
    a = np.array([[i, val] for i, val in enumerate(action.numpy())])

    with tf.GradientTape() as tape:
        prediction = dqn(state)
        out = tf.gather_nd(prediction["q_values"], tf.constant(a))
        loss = loss_func(target, out)
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
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.Adam()
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
    # print("test before training: ")
    # manager.test(test_steps, do_print=True, render=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)

        # create and batch tf datasets
        data_dict = dict_to_dataset(sample_dict, batch_size=optim_batch_size)

        loss = 0
        for s, a, r, sn, nd in data_dict:
            q_target = r + gamma * agent.max_q(sn)
            loss += train(agent.model, s, a, q_target, optimizer, loss_function)


        # set new weights
        new_weights = agent.model.get_weights()
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)

        print(f"epoch ::: {e}  loss ::: {np.round(loss,4)}   avg env steps ::: {np.mean(time_steps)}")

        # you can also alter your managers parameters
        # manager.set_epsilon(epsilon=0.99)

        # if e % saving_after == 0:
        #     manager.save_model(saving_path, e)


    #manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
