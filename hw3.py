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
        self.middle_layer_neurons = 28
        self.second_layer_neurons = 16

        self.layer_list = [
            tf.keras.layers.Dense(self.middle_layer_neurons, activation=tf.nn.leaky_relu, input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(2, use_bias=False, activation='tanh')]


    def call(self, state):

        for layer in self.layer_list:
            state = layer(state)
        
        return state


class ValueEstimator(tf.keras.Model):

    def __init__(self, state_size, batch_size):

        super(ValueEstimator, self).__init__()
        self.state_size = state_size
        self.second_layer_neurons = 16

        self.layer_list = [
            tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.nn.leaky_relu, input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(1)]


    def call(self, state):

        for layer in self.layer_list:
            state = layer(state)
        
        return state


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



def train_pi(action, state, r_sum, value, opt):
    
    with tf.GradientTape() as tape:

        
        state = np.array(state)
        state = np.reshape(state, (1,8))

        print("State: ", state)

        mue = tf.cast(agent.model.pi_network(state), tf.float64)
        # dist = tf.contrib.distributions.Normal(mue, agent.model.sigma)
        # prob = dist.pdf(action)
        
        tfd = tfp.distributions
        dist = tfd.Normal(loc=mue, scale=tf.cast(agent.model.sigma, tf.float64))
        prob = dist.cdf(action)

        target = tf.math.log(prob)

        print("r_sum: ", r_sum)
        print("value: ", value)
        print("target: ", target)

        factor = tf.cast((r_sum - tf.cast(tf.squeeze(value), tf.float64)), tf.float64)

        print("Factor: ", factor)

        gradients = tf.cast(tape.gradient(target, agent.model.pi_network.trainable_variables), tf.float64)

        print("Gradients: ", gradients)
        factored_gradients = factor * gradients
        opt.apply_gradients(zip(-1 * gradients, agent.model.pi_network.trainable_variables))

    return loss

def train_v(agent, r_sum, state, opt):

    with tf.GradientTape() as tape:

        val = agent.model.value_network(tf.expand_dims(state, axis=0))

        loss = tf.keras.losses.MSE(r_sum, val)
        gradients = tape.gradient(loss, agent.model.value_network.trainable_variables)
        opt.apply_gradients(zip(gradients, agent.model.value_network.trainable_variables))

    return loss


if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    batch_size = 1
    state_size = 8

    model_kwargs = {
        "batch_size": batch_size,
        "state_size": state_size,
        "sigma1": 0.1,
        "sigma2": 0.1
    }

    kwargs = {
        "model": ModelWrapper,
        "model_kwargs": model_kwargs,
        "returns": ['monte_carlo'], 
        "environment": 'LunarLanderContinuous-v2',
        "num_parallel": 1,
        "total_steps": 50, # how many total steps we do
        "num_steps": 50,
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

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        def get_end(data):
            for i, d in enumerate(data):
                if d == 0:
                    return i, True
            return len(data)-1, False
        
        end, terminal = get_end(sample_dict['not_done'])

        #entries = ['state', 'action', 'reward', 'state_new', 'monte_carlo']

        #data = [data_dict[val] for val in entries]

        states = sample_dict['state'][:end]
        actions = sample_dict['action'][:end]
        new_states = sample_dict['state_new'][:end]
        mc_rewards = sample_dict['monte_carlo'][:end]
        
        if terminal: # TODO think about this
            R = agent.model(states[end])['value_estimate']
            mc_rewards.apend(gamma ** end * R)

        r_sum = tf.reduce_sum(mc_rewards)
        old_mcr = 0
        for s, a, sn, mc_r in zip(states, actions, new_states, mc_rewards): 
            r_sum -= old_mcr
            value = agent.model(tf.expand_dims(s, axis=0))['value_estimate']
            old_mcr = mc_r

            loss_pi = train_pi(a, s, r_sum, value, None)
            loss_v = train_v(agent, r_sum, s, optimizer)

            print("value loss: ", loss_v)
            print("pi loss: ", loss_pi)

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
