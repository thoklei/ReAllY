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
            tf.keras.layers.Dense(self.middle_layer_neurons, activation='tanh', input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.second_layer_neurons, activation='tanh'),
            tf.keras.layers.Dense(2, use_bias=False, activation='tanh')]


    def call(self, state):

        for layer in self.layer_list:
            state = layer(state)
        
        return state


class ValueEstimator(tf.keras.Sequential):

    def __init__(self, state_size, batch_size):

        super(ValueEstimator, self).__init__()
        self.state_size = state_size
        self.second_layer_neurons = 16

        self.add(tf.keras.layers.Dense(self.second_layer_neurons, activation='tanh', input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.Dense(1, use_bias=False))


    # def call(self, state):

    #     for layer in self.layer_list:
    #         state = layer(state)
        
    #     return state


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

        mue = tf.cast(agent.model.pi_network(state), tf.float64)
        
        tfd = tfp.distributions
        dist = tfd.Normal(loc=mue, scale=tf.cast(agent.model.sigma, tf.float64))
        prob = dist.cdf(action)

        target = tf.math.log(prob)

        factor = r_sum - tf.cast(tf.squeeze(value), tf.float64)

        gradients = tape.gradient([factor * target], agent.model.pi_network.trainable_variables)

        opt.apply_gradients(zip(-1 * gradients, agent.model.pi_network.trainable_variables))


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
    gamma = 0.99

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
        "total_steps": 400, # how many total steps we do
        "num_steps": 400,
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
    epochs = 100
    optim_batch_size = 1
    saving_after = 10

    
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
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps", "reward"]
    )

    # initial testing:
    print("Establishing baseline.")
   # manager.test(test_steps, test_episodes=3, do_print=True, render=True)

    print("Training the agent.")

    # get initial agent
    agent = manager.get_agent()
    #agent.model.build(optim_batch_size)

    agent.model.build((1,8))
    for e in range(epochs):

        # data = manager.get_data()
        # manager.store_in_buffer(data)

        sample_dict = manager.sample(1, from_buffer=False) # 

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        def get_end(data):
            for i, d in enumerate(data):
                if d == 0:
                    return i+1, True
            return len(data)-1, False

        #print("Len not_done: ", len(sample_dict['not_done']), " Len states: ", len(sample_dict['state'])) 
        
        end, terminal = get_end(sample_dict['not_done'])


        states = sample_dict['state'][:end]
        #print("length of states: ", len(states))
        #print("end: ", end, terminal)
        actions = sample_dict['action'][:end]
        new_states = sample_dict['state_new'][:end]
        rewards = sample_dict['reward'][:end]
        print(np.min(sample_dict['reward']), " max: ", np.max(sample_dict['reward']))
        print("nachher:", np.min(rewards), " max: ", np.max(rewards))
        mc_rewards = [ sum([gamma**j*r for j,r in enumerate(rewards[i:])]) for i in range(len(rewards)) ]
        # mc_rewards = sample_dict['monte_carlo'][:end]

        # print("Normal rewards: ", rewards)
        # print("MC rewards: ", mc_rewards)
        
        if terminal:
            R = agent.model(np.reshape(states[end-1], (1,8)))['value_estimate']
            mc_rewards.append(gamma ** (end-1) * R)

        loss = 0

        stormtrooper = tf.keras.models.clone_model(agent.model.value_network)

        it = 1
        for s, a, sn, mc_r in zip(states, actions, new_states, mc_rewards): 
            #print("action:", a)

            value = stormtrooper(tf.expand_dims(s, axis=0))

            train_pi(a, s, mc_r, value, optimizer)
            loss_v = train_v(agent, mc_r, s, optimizer)

            # old average * (n-1)/n + new value /n
            loss = loss * (it-1)/it + loss_v / it
            it += 1

        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        if e % 10 == 0:
            time_steps, rewards = manager.test(test_steps, test_episodes=10, render=True, evaluation_measure="time_and_reward")
            print("should be 10", len(rewards))

            manager.update_aggregator(loss=loss, time_steps=time_steps, reward=rewards)

            print(f"epoch ::: {e}  loss ::: {loss}   reward ::: {np.sum(rewards)}   avg env steps ::: {np.mean(time_steps)}")

        # else:
        #     time_steps, rewards = manager.test(test_steps, render=False, evaluation_measure="time_and_reward")
        #     print("should be 100", len(rewards))
       
       
    print("Done.")
    print("Testing optimized agent.")

    manager.test(100, test_episodes=10, render=True, do_print=True)
