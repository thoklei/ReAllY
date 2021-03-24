import os
import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import ray
from really import SampleManager
from gridworlds import GridWorld
from really.utils import (
    dict_to_dataset, dict_to_dict_of_datasets
)

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.random.set_seed(1234)
DEBUG = False

def dprint(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)

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

        self.reg = tf.keras.regularizers.L2(l2=0.05) 

        self.add(tf.keras.layers.Dense(self.middle_layer_neurons, input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.LeakyReLU(alpha=0.1, activity_regularizer=self.reg))
        self.add(tf.keras.layers.Dense(self.second_layer_neurons, activity_regularizer=self.reg))
        self.add(tf.keras.layers.LeakyReLU(alpha=0.1, activity_regularizer=self.reg))
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
        self.middle_layer_neurons = 48
        self.second_layer_neurons = 32

        self.add(tf.keras.layers.Dense(self.middle_layer_neurons, input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.LeakyReLU(alpha=0.05))
        self.add(tf.keras.layers.Dense(self.second_layer_neurons))
        self.add(tf.keras.layers.LeakyReLU(alpha=0.05))
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
        self.sigma = tf.constant(np.array([sigma1, sigma2]), dtype=tf.float32)


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


def train_pi(pi, old_pi, action, state, value, value_estimate, opt, entropy_coefficient, clip_param):
    """
    Trains the Policy Network by doing gradient ascent on the clipped PPO-objective.

    pi             :    the policy network to be trained
    old_pi         :    the old version of the policy net, to calculate KL divergence
    action         :    the action that was taken, tensor of shape (batch_size,2)
    state          :    the state, tensor of shape (batch_size,state_size)
    value          :    the observed value, i.e. r + gamma * value_net(next_state), tensor of shape (batch_size,)
    value_estimate :    the estimated value, i.e. output of value_net(state), tensor of shape (batch_size,)
    opt            :    the optimizer
    """

    # casting everything to float32
    state = tf.cast(state, dtype=tf.float32)
    value = np.array(value, dtype=np.float32)
    action = tf.cast(action, dtype=tf.float32)
    value_estimate = tf.cast(value_estimate, tf.float32)

    value_estimate = tf.squeeze(value_estimate)

    # calculate advantage as difference between 
    # observed value for this action and current state value estimate
    # TODO make this more flexible to allow for different advantage estimates, e.g. GAE
    advantage = value - value_estimate

    def get_prob_dist(_net, _state):
        return tfd.Normal(loc=_net(_state), scale=agent.model.sigma)


    with tf.GradientTape() as tape:
    
        old_pd = get_prob_dist(old_pi, state)
        old_prob = old_pd.prob(action)

        new_pd = get_prob_dist(pi, state)
        new_prob = new_pd.prob(action)

        # calculate entropy - doesn't make much sense yet because fixed sigma, so will be constant
        # for each train loop. Will make sense though as soon as we output sigma as well.
        entropy = new_pd.entropy()
        dprint("Entropy:", tf.reduce_mean(entropy))
        entropy_penalty = -1 * entropy_coefficient * tf.reduce_mean(entropy)

        # calculate ratio
        ratio = tf.exp(tf.math.log(new_prob) - tf.math.log(old_prob)) # short for pnew / pold
        ratio = tf.transpose(ratio)

        # calculate clipped and unclipped surrogate objectives
        surrogate1 = ratio * advantage
        surrogate2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        # collect surrogate loss across batch
        surrogate = -1 * tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # calculate regularizer loss
        regularizer_loss = sum(pi.losses)

        for n, l in zip(["surrogate_loss:", "entropy_penalty:", "regularizer_loss:"],[surrogate, entropy_penalty, regularizer_loss]):
            dprint(n,l)

        # calculate final loss
        loss = surrogate + entropy_penalty + regularizer_loss

    # calculate and apply gradients
    gradients = tape.gradient(loss, pi.trainable_variables)
    old = tf.keras.models.clone_model(pi)
    dprint("Gradients_mean:", np.mean([tf.norm(g) for g in gradients]), "Gradients Max:", max([tf.norm(g) for g in gradients]))
    clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
    opt.apply_gradients(zip(clipped_gradients, pi.trainable_variables))

    return loss, old, pi


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
        "sigma1": 0.6,
        "sigma2": 0.3
    }

    kwargs = {
        "model": ModelWrapper,
        "model_kwargs": model_kwargs,
        "environment": 'LunarLanderContinuous-v2',
        "num_parallel": 2,
        "total_steps": 1200, # how many total steps to do
        "num_steps": 300,
        #"num_episodes": 10, # we prefer few full episodes (that get to the reward) over many short ones
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
    epochs = 60
    saving_after = 10
    sample_size = 1000
    gamma = 0.96

    actor_optimizer = tf.keras.optimizers.Adam() 
    critic_optimizer = tf.keras.optimizers.Adam()
    ########################
    ## </Hyperparameters> ##
    ########################

    # initialize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps", "reward"]
    )

    # initial testing:
    print("Establishing baseline.")
    #manager.test(test_steps, test_episodes=3, do_print=True, render=True)

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
        vloss = 0
        ploss = 0
        old_pi = tf.keras.models.clone_model(agent.model.pi_network)
        pi = agent.model.pi_network
        for s, a, sn, r, nd in zip(data_dict['state'], data_dict['action'], data_dict['state_new'], data_dict['reward'], data_dict['not_done']): 

            # calculate true value
            value = r.numpy() + nd.numpy() * gamma * tf.squeeze(agent.model.value_network(sn)).numpy()

            # obtain current value estimate
            value_estimate = agent.model.value_network(s)

            # train policy net
            loss_p, old_pi, pi = train_pi(pi, old_pi, a, s, value, value_estimate, actor_optimizer, 0.001, 0.2)

            # train value net
            loss_v = np.mean(train_v(agent, s, value, critic_optimizer))

            # averagelosses according to old average * (n-1)/n + new value /n
            vloss = vloss * (it-1)/it + loss_v / it
            ploss = ploss * (it-1)/it + loss_p / it
            it += 1

        # Decaying sigma, but not smaller than 0.05
        manager.kwargs["model_kwargs"]["sigma1"] = max(manager.kwargs["model_kwargs"]["sigma1"] * 0.99, 0.05)
        manager.kwargs["model_kwargs"]["sigma2"] = max(manager.kwargs["model_kwargs"]["sigma2"] * 0.99, 0.05)

        agent.model.pi_network = pi
        # update with new weights
        new_weights = agent.model.get_weights()
        
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # test agent, render every 5th episode
        time_steps, rewards = manager.test(test_steps, test_episodes=10, render=(e % 5 == 0), evaluation_measure="time_and_reward")
        
        # update aggregator
        manager.update_aggregator(loss=vloss+ploss, time_steps=time_steps, reward=rewards)

        print(f"epoch: {e:02}    value_loss: {vloss:05.3f}    policy_loss: {ploss:.3f}    reward: {np.mean(rewards):.3f}    avg env steps: {np.mean(time_steps):.2f}")
       
       
    print("Done.")
    print("Testing optimized agent.")

    manager.test(250, test_episodes=10, render=True, do_print=True)
