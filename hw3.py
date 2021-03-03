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
            tf.keras.layers.Dense(self.middle_layer_neurons, activation=tf.nn.leaky_relu, input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(n_actions, use_bias=False)]


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

    with tf.GradientTape() as tape:

        out = agent.q_val(state, action)
        loss = loss_func(target, out)
        gradients = tape.gradient(loss, agent.model.trainable_variables)
        optim.apply_gradients(zip(gradients, agent.model.trainable_variables))

    return loss



# @tf.function
def train(dqn, state, action, target, optim, loss_func):
    """
    Trains the deep Q-Network.

    dqn: the network
    state: the state in which the action was taken
    action: the taken action
    target: the q-values according to the Watkins-Formula
    optim: the optimizer to be used (e.g. Adam)
    loss_func: the loss function to be used (e.g. MSE)
    """

    a = np.array([[i, val] for i, val in enumerate(action.numpy())])

    with tf.GradientTape() as tape:
        prediction = dqn(state)
        # we want to run backprop only on the actions we took.
        #relevant_qvals = tf.gather_nd(prediction["q_values"], action, 1)
        #loss = loss_func(target, relevant_qvals)

        out = tf.gather_nd(prediction["q_values"], tf.constant(a))
        loss = loss_func(target, out)

        gradients = tape.gradient(loss, dqn.trainable_variables)
        optim.apply_gradients(zip(gradients, dqn.trainable_variables))

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
        "environment": 'LunarLander-v2',
        "num_parallel": 4,
        "total_steps": 400,
        "action_sampling_type": "epsilon_greedy",
        "epsilon": 0.8
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
    epochs = 25
    sample_size = 4000
    optim_batch_size = 32
    saving_after = 10

    gamma = 0.9
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9) #Adam(learning_rate=learning_rate) #SGD(learning_rate, momentum=0.8)
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
    #agent.model.build(optim_batch_size)

    for e in range(epochs):

        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size, from_buffer=False)

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        
        loss = 0.0
        for state, action, reward, state_next, nd in zip(data_dict['state'], data_dict['action'], data_dict['reward'], data_dict['state_new'], data_dict['not_done']):

            q_target = tf.cast(reward,tf.float64) + (tf.cast(nd, tf.float64) * tf.cast(gamma * agent.max_q(state_next), tf.float64))
            # use backpropagation and count up the losses
            loss += train_new(agent, state, action, q_target, optimizer, loss_function)


        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps, render=False)
        manager.update_aggregator(loss=loss, time_steps=time_steps)

        print(f"epoch ::: {e}  loss ::: {loss.numpy()}   avg env steps ::: {np.mean(time_steps)}")

        # Annealing epsilon
        if e % 5 == 0: 
            new_epsilon = 0.9 * manager.kwargs['epsilon']
            manager.set_epsilon(new_epsilon)

        # if e % saving_after == 0:
        #     manager.save_model(saving_path, e)

    # manager.load_model(saving_path)
    print("Done.")
    print("Testing optimized agent.")

    manager.test(100, test_episodes=10, render=True, do_print=True)
