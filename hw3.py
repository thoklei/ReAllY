import os
import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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

        self.reg = tf.keras.regularizers.L2(l2=0.01)

        self.layer_list = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation='tanh', kernel_regularizer='l2', input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.second_layer_neurons, activation='tanh', kernel_regularizer='l2'),
            tf.keras.layers.Dense(2, use_bias=False, activation='tanh', kernel_regularizer='l2')]


    def call(self, state):

        for layer in self.layer_list:
            state = layer(state)
        
        return state


class ValueEstimator(tf.keras.Sequential):

    def __init__(self, state_size, batch_size):

        super(ValueEstimator, self).__init__()
        self.state_size = state_size
        self.middle_layer_neurons = 32
        self.second_layer_neurons = 16

        self.add(tf.keras.layers.Dense(self.middle_layer_neurons, activation='tanh', input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.Dense(self.second_layer_neurons, activation='tanh'))
        self.add(tf.keras.layers.Dense(1, use_bias=False))


class ModelWrapper(tf.keras.Model):
        
    def __init__(self, state_size, batch_size, sigma1=0.1, sigma2=0.1):
        super(ModelWrapper, self).__init__()
        self.value_network = ValueEstimator(state_size, batch_size)
        self.pi_network = Pi(state_size, batch_size)
        self.sigma = tf.constant(np.array([sigma1, sigma2]))


    # @tf.function
    def call(self, x):

        # obtain mu of value network
        mu = self.pi_network(x)

        # get an value estimate from the value network
        value = self.value_network(tf.identity(x))

        output = {}
        output["mu"] = mu
        output["sigma"] = self.sigma
        output["value_estimate"] = value

        return output



def train_pi(action, state, value, value_estimate, opt):
    """
    Trains the Policy Network.

    action = the action that was taken, tensor of shape (32,2)
    state = the state, tensor of shape (32,8)
    value = the observed value, i.e. monte carlo rewards, tensor of shape (32,)
    value_estimate = the estimated value, i.e. output of value network, tensor of shape (32,)
    opt = the optimizer
    """
    factor = value - tf.cast(tf.squeeze(value_estimate), tf.float64)

    with tf.GradientTape() as tape:

        mue = tf.cast(agent.model.pi_network(state), tf.float64)
        
        tfd = tfp.distributions
        dist = tfd.Normal(loc=mue, scale=tf.cast(agent.model.sigma, tf.float64))
        prob = dist.cdf(action)

        target = tf.math.log(prob)

        #print("Factor:", factor)
        #print("target: ", target)

        weighted = tf.math.multiply(tf.transpose(target), factor)

        #print("weighted: ", weighted)

    gradients = tape.gradient(weighted, agent.model.pi_network.trainable_variables)

    opt.apply_gradients(zip(-1 * gradients, agent.model.pi_network.trainable_variables))


def train_v(agent, r_sum, state, opt):

    with tf.GradientTape() as tape:

        val = agent.model.value_network(state)

        loss = tf.keras.losses.MSE(r_sum, val)
    gradients = tape.gradient(loss, agent.model.value_network.trainable_variables)
    opt.apply_gradients(zip(gradients, agent.model.value_network.trainable_variables))

    return loss


if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    batch_size = 32
    state_size = 8
    gamma = 0.99


    model_kwargs = {
        "batch_size": batch_size,
        "state_size": state_size,
        "sigma1": 1.0,
        "sigma2": 0.6
    }

    kwargs = {
        "model": ModelWrapper,
        "model_kwargs": model_kwargs,
        "returns": ['monte_carlo'], 
        "environment": 'LunarLanderContinuous-v2',
        "num_parallel": 2,
        "total_steps": 1000, # how many total steps we do
        "num_steps": 1000,
        "action_sampling_type": "continuous_normal_diagonal",
    }

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)


    #######################
    ## <Hyperparameters> ##
    #######################
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 42
    test_steps = 250
    epochs = 250
    saving_after = 10
    sample_size = 1000

    
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam() #SGD(learning_rate, momentum=0.8)
    loss_function = tf.keras.losses.MSE

    ########################
    ## </Hyperparameters> ##
    ########################

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    # manager.initilize_buffer(buffer_size, optim_keys)

    # # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps", "reward"]
    )

    # initial testing:
    print("Establishing baseline.")
   # manager.test(test_steps, test_episodes=3, do_print=True, render=True)

    print("Training the agent.")

    # get initial agent
    agent = manager.get_agent()
    #agent.model.build(optim_batch_size)

    agent.model.build((batch_size,8))
    for e in range(epochs):
        print("Working on epoch",e)

        # data = manager.get_data()
        # manager.store_in_buffer(data)

        sample_dict = manager.sample(sample_size, from_buffer=False) # 

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=batch_size)

        it = 1
        loss = 0
        for s, a, sn, mc_r in zip(data_dict['state'], data_dict['action'], data_dict['state_new'], data_dict['monte_carlo']): 
            
            # print("state:", s)

            value = agent.model.value_network(tf.expand_dims(s, axis=0))

            train_pi(a, s, mc_r, value, optimizer)
            loss_v = np.mean(train_v(agent, mc_r, s, optimizer))

            # old average * (n-1)/n + new value /n
            loss = loss * (it-1)/it + loss_v / it
            it += 1

        print("Sigma: ", agent.model.sigma)
        manager.kwargs["model_kwargs"]["sigma1"] = max(manager.kwargs["model_kwargs"]["sigma1"] * 0.99, 0.1)
        manager.kwargs["model_kwargs"]["sigma2"] = max(manager.kwargs["model_kwargs"]["sigma2"] * 0.98, 0.05)

        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        if (e+1) % 10 == 0:
            time_steps, rewards = manager.test(test_steps, test_episodes=10, render=True, evaluation_measure="time_and_reward")

            manager.update_aggregator(loss=loss, time_steps=time_steps, reward=rewards)

            print(f"epoch ::: {e}  loss ::: {loss}   reward ::: {np.sum(rewards)}   avg env steps ::: {np.mean(time_steps)}")

            # annealing sigma
            
        # else:
        #     time_steps, rewards = manager.test(test_steps, render=False, evaluation_measure="time_and_reward")
        #     print("should be 100", len(rewards))
       
       
    print("Done.")
    print("Testing optimized agent.")

    manager.test(100, test_episodes=10, render=True, do_print=True)
