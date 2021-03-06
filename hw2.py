import os
import logging
import gym
import numpy as np
import tensorflow as tf
import ray
from really import SampleManager
from gridworlds import GridWorld
from really.utils import (
    dict_to_dataset, dict_to_dict_of_datasets
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
        self.middle_layer_neurons = 32
        self.second_layer_neurons = 16

        self.layer_list = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation=tf.nn.leaky_relu, input_shape=(batch_size, state_size), kernel_regularizer='l2'),
            tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.nn.leaky_relu, kernel_regularizer='l2'),
            tf.keras.layers.Dense(n_actions, use_bias=False, kernel_regularizer='l2')]


    @tf.function
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


def train_new(agent, state, action, target, optim, loss_func):
    """
    Trains the agent to output correct q-values for a state-action pair.

    agent = the agent to be trained
    state = the environment state to be trained on
    action = the action to be trained on 
    target = the immediate reward + maximum q-value of the succesive state
    optim = the optimizer to be used
    loss_func = the loss function to be used (MSE)
    """        

    with tf.GradientTape() as tape:

        out = agent.q_val(state, action)
        loss = loss_func(target, out)
        gradients = tape.gradient(loss, agent.model.trainable_variables)
        optim.apply_gradients(zip(gradients, agent.model.trainable_variables))

    return loss


if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    model_kwargs = {
        "batch_size": 32,
        "state_size": 4,
        "n_actions": 2
    }

    kwargs = {
        "model": DQN,
        "model_kwargs": model_kwargs,
        "environment": 'CartPole-v0',
        "num_parallel": 4,
        "total_steps": 1500,
        "action_sampling_type": "epsilon_greedy",
        "epsilon": 0.95
    }

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)


    #######################
    ## <Hyperparameters> ##
    #######################
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 30000
    test_steps = 250
    epochs = 30
    sample_size = 4000
    optim_batch_size = 32
    saving_after = 10

    gamma = 0.9
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.MSE

    ########################
    ## </Hyperparameters> ##
    ########################

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("Establishing baseline.")
    manager.test(test_steps, test_episodes=10, do_print=True, render=True)

    print("Training the agent.")

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        # sampling fresh trajectories seems to work better, should not be necessary though
        sample_dict = manager.sample(sample_size, from_buffer=False) 

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        
        loss = 0.0
        for state, action, reward, state_next, nd in zip(data_dict['state'], data_dict['action'], data_dict['reward'], data_dict['state_new'], data_dict['not_done']):

            q_target = tf.cast(reward,tf.float64) + (tf.cast(nd, tf.float64) * tf.cast(gamma * agent.max_q(state_next), tf.float64))
            
            # use backpropagation and sum up the losses
            loss += train_new(agent, state, action, q_target, optimizer, loss_function)


        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps, test_episodes=5, render=False)
        manager.update_aggregator(loss=loss, time_steps=time_steps)

        print(f"epoch ::: {e}  loss ::: {loss.numpy()}   avg env steps ::: {np.mean(time_steps)}")

        # Annealing epsilon
        if (e+1) % 5 == 0: 
            new_epsilon = 0.85 * manager.kwargs['epsilon']
            manager.set_epsilon(new_epsilon)
            print("New Epsilon: ", new_epsilon)
        

    print("Done!")
    print("Testing optimized agent...")

    manager.set_epsilon(0.0)
    manager.test(test_steps, test_episodes=10, render=True, do_print=True)
