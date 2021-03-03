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


class Pi(tf.keras.Model):

    def __init__(self, state_size, batch_size):

        super(Pi, self).__init__()
        self.state_size = state_size
        self.middle_layer_neurons = 32
        self.second_layer_neurons = 16

        self.layer_list_pi = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation=tf.nn.leaky_relu, input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(2)]

        self.layer_list_v = [
            tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.nn.leaky_relu, input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(1)]


    def call(self, state):

        state2 = state

        for layer in self.layer_list_pi:
            state = layer(state)

        for layer in self.layer_list_v:
            state2 = layer(state2)    

        output = {}
        output["mu"] = state
        output["sigma"] = tf.constant(0.1)
        output["value_estimate"] = state2

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
        "batch_size": 1,
        "state_size": 8,
    }

    kwargs = {
        "model": Pi,
        "model_kwargs": model_kwargs,
        "returns": ['monte_carlo'], 
        "environment": 'LunarLanderContinuous-v2',
        "num_parallel": 1,
        "total_steps": 20, # how many total steps we do
        "num_steps": 20,
        "action_sampling_type": "continous_normal_diagonal",
        "epsilon": 0.9
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
    optim_batch_size = 1
    saving_after = 10

    gamma = 0.9
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam() #SGD(learning_rate, momentum=0.8)
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
   # manager.test(test_steps, test_episodes=3, do_print=True, render=True)

    print("Training the agent.")

    # get initial agent
    agent = manager.get_agent()
    #agent.model.build(optim_batch_size)

    for e in range(epochs):

        # data = manager.get_data()
        # manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(1, from_buffer=False) # 

        #print(sample_dict)

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        step = 0
        for state, action, reward, state_next, nd, mc in zip(data_dict['state'], data_dict['action'], data_dict['reward'], data_dict['state_new'], data_dict['not_done'], data_dict['monte_carlo']):
            print("MC rewards: ", mc)
        data_dict = None

        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps, render=False)
        manager.update_aggregator(loss=step, time_steps=time_steps)

        #print(f"epoch ::: {e}  loss ::: {loss.numpy()}   avg env steps ::: {np.mean(time_steps)}")

        # Annealing epsilon
        # if e % 5 == 0: 
        #     new_epsilon = 0.9 * manager.kwargs['epsilon']
        #     manager.set_epsilon(new_epsilon)

        # if e % saving_after == 0:
        #     manager.save_model(saving_path, e)

    # manager.load_model(saving_path)
    print("Done.")
    print("Testing optimized agent.")

    manager.test(100, test_episodes=10, render=True, do_print=True)
