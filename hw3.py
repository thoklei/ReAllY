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

# Policy Network (Actor)
class Pi(tf.keras.Sequential):

    def __init__(self, state_size, batch_size):
        """
        Constructor for policy net.

        state_size :    state space dimensionality
        batch_size :    batchsize to build model with
        """

        super(Pi, self).__init__()
        self.state_size = state_size
        self.middle_layer_neurons = 32
        self.second_layer_neurons = 16

        self.reg = tf.keras.regularizers.L2(l2=0.1)

        self.add(tf.keras.layers.Dense(self.middle_layer_neurons, activity_regularizer=self.reg, activation=tf.keras.layers.LeakyReLU(alpha=0.05), input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.Dense(self.second_layer_neurons, activity_regularizer=self.reg, activation=tf.keras.layers.LeakyReLU(alpha=0.05)))
        self.add(tf.keras.layers.Dense(2, use_bias=False, activity_regularizer=self.reg, activation='tanh'))


# Value Network (Critic)
class ValueEstimator(tf.keras.Sequential):

    def __init__(self, state_size, batch_size):
        """
        Constructor for value net.

        state_size :    state space dimensionality
        batch_size :    batchsize to build model with     
        """

        super(ValueEstimator, self).__init__()
        self.state_size = state_size
        self.middle_layer_neurons = 32
        self.second_layer_neurons = 16

        self.add(tf.keras.layers.Dense(self.middle_layer_neurons, activation=tf.keras.layers.LeakyReLU(alpha=0.05), input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.Dense(self.second_layer_neurons, activation=tf.keras.layers.LeakyReLU(alpha=0.05)))
        self.add(tf.keras.layers.Dense(1, use_bias=False))


# full A2C model
class ModelWrapper(tf.keras.Model):
        
    def __init__(self, state_size, batch_size, sigma1=1.0, sigma2=0.7):
        """
        Constructor for A2C.

        state_size :    state space dimensionality
        batch_size :    batchsize to build model with
        sigma1     :    sigma for action1 (main thruster)
        sigma2     :    sigma for action2 (side thrusters)
        """
        super(ModelWrapper, self).__init__()
        self.value_network = ValueEstimator(state_size, batch_size)
        self.pi_network = Pi(state_size, batch_size)
        self.sigma = tf.constant(np.array([sigma1, sigma2]))


    def call(self, x):

        # obtain mu of policy network
        mu = self.pi_network(x)

        # get a value estimate from the value network
        value = self.value_network(x)

        output = {}
        output["mu"] = mu
        output["sigma"] = self.sigma
        output["value_estimate"] = value

        return output


def train_pi(agent, action, state, value, value_estimate, opt):
    """
    Trains the Policy Network.

    action         :    the action that was taken, tensor of shape (batch_size,2)
    state          :    the state, tensor of shape (batch_size,state_size)
    value          :    the observed value, i.e. r + gamma * value_net(next_state), tensor of shape (batch_size,)
    value_estimate :    the estimated value, i.e. output of value_net(state), tensor of shape (batch_size,)
    opt            :    the optimizer
    """
    
    value = np.array(value)
    value_estimate = tf.squeeze(value_estimate)

    # calculate advantage as difference between 
    # observed value for this action and current state value estimate
    advantage = value - tf.cast(value_estimate, tf.float64)

    with tf.GradientTape() as tape:

        # obtain mue for action from actor
        mue = tf.cast(agent.model.pi_network(state), tf.float64)
        
        # calculate probability of action
        tfd = tfp.distributions
        dist = tfd.Normal(loc=mue, scale=tf.cast(agent.model.sigma, tf.float64))
        prob = dist.prob(action)

        # calculate target as negative log prob of action
        target = -1 * tf.math.log(prob)

        # collect regularizer losses
        regularizer_loss = sum(tf.cast(agent.model.pi_network.losses, tf.float64))

        # calculate objective as advantage-weighted target
        objective = tf.math.multiply(tf.transpose(target), advantage) 
        
        # combine objective with regularizer loss
        objective = objective + regularizer_loss

    # calculate and apply gradients
    gradients = tape.gradient(objective, agent.model.pi_network.trainable_variables)
    opt.apply_gradients(zip(gradients, agent.model.pi_network.trainable_variables))


def train_v(agent, state, true_value, opt):
    """
    Trains the Value Network.

    agent      :    the agent
    state      :    the state, tensor of shape (batch_size, state_size)
    true_value :    the observed value, i.e. r + gamma * value_net(next_state), tensor of shape (batch_size,)
    opt        :    the optimizer
    """

    with tf.GradientTape() as tape:

        # obtain current value estimate
        val = agent.model.value_network(state)

        # calculate loss as MSE between current estimate and true value
        loss = tf.keras.losses.MSE(true_value, tf.squeeze(val)) 

    # calculate and apply gradients
    gradients = tape.gradient(loss, agent.model.value_network.trainable_variables)
    opt.apply_gradients(zip(gradients, agent.model.value_network.trainable_variables))

    return loss


if __name__ == "__main__":

    if not os.path.exists("./logging/"):
        os.makedirs("logging")

    batch_size = 32
    state_size = 8
    
    model_kwargs = {
        "batch_size": batch_size,
        "state_size": state_size,
        "sigma1": 0.7,
        "sigma2": 0.4
    }

    kwargs = {
        "model": ModelWrapper,
        "model_kwargs": model_kwargs,
        "environment": 'LunarLanderContinuous-v2',
        "num_parallel": 2,
        "total_steps": 1000, # how many total steps to do
        "num_episodes": 10, # we prefer few full episodes (that get to the reward) over many short ones
        "action_sampling_type": "continuous_normal_diagonal",
    }

    # initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    #######################
    ## <Hyperparameters> ##
    #######################

    saving_path = os.getcwd() + "/progress_test"

    test_steps = 500
    epochs = 250
    saving_after = 10
    sample_size = 1000
    gamma = 0.9

    optimizer = tf.keras.optimizers.Adam() 

    ########################
    ## </Hyperparameters> ##
    ########################

    # initialize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps", "reward"]
    )

    # initial testing:
    print("Establishing baseline.")
    manager.test(test_steps, test_episodes=3, do_print=True, render=True)

    print("Training the agent.")

    # get initial agent
    agent = manager.get_agent()
    agent.model.build((batch_size,state_size))

    # main training loop
    for e in range(epochs):

        # sample trajectories from current policy
        sample_dict = manager.sample(sample_size, from_buffer=False)
        
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=batch_size)

        it = 1
        loss = 0
        for s, a, sn, r, nd in zip(data_dict['state'], data_dict['action'], data_dict['state_new'], data_dict['reward'], data_dict['not_done']): 

            # calculate true value
            value = r.numpy() + nd.numpy() * gamma * tf.squeeze(agent.model.value_network(sn)).numpy()

            # obtain current value estimate
            value_estimate = agent.model.value_network(s)

            # train policy net
            train_pi(agent, a, s, value, value_estimate, optimizer)

            # train value net
            loss_v = np.mean(train_v(agent, s, value, optimizer))

            # averagelosses according to old average * (n-1)/n + new value /n
            loss = loss * (it-1)/it + loss_v / it
            it += 1

        # Decaying sigma, but not smaller than 0.05
        manager.kwargs["model_kwargs"]["sigma1"] = max(manager.kwargs["model_kwargs"]["sigma1"] * 0.99, 0.05)
        manager.kwargs["model_kwargs"]["sigma2"] = max(manager.kwargs["model_kwargs"]["sigma2"] * 0.99, 0.05)

        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # test agent, render every 5th episode
        time_steps, rewards = manager.test(test_steps, test_episodes=10, render=(e % 5 == 0), evaluation_measure="time_and_reward")
        
        # update aggregator
        manager.update_aggregator(loss=loss, time_steps=time_steps, reward=rewards)

        print(f"epoch: {e}    loss: {loss}    reward: {np.mean(rewards)}    avg env steps: {np.mean(time_steps)}")
       
       
    print("Done.")
    print("Testing optimized agent.")

    manager.test(250, test_episodes=10, render=True, do_print=True)
