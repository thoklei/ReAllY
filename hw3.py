import os
import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import ray
from really import SampleManager
from gridworlds import GridWorld
from time import time_ns
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
            tf.keras.layers.Dense(self.middle_layer_neurons, activation="tanh", input_shape=(batch_size, state_size)),
            tf.keras.layers.Dense(self.second_layer_neurons, activation="tanh"),
            tf.keras.layers.Dense(2, use_bias=False, activation='tanh')]

    @tf.function
    def call(self, state):
        for layer in self.layer_list:
            state = layer(state)
        return state


class ValueEstimator(tf.keras.Sequential):

    def __init__(self, state_size, batch_size):
        super(ValueEstimator, self).__init__()
        self.state_size = state_size
        self.second_layer_neurons = 16

        self.add(
            tf.keras.layers.Dense(self.second_layer_neurons, activation="relu", input_shape=(batch_size, state_size)))
        self.add(tf.keras.layers.Dense(32, activation="relu"))
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
        self.sigma = tf.constant([sigma1, sigma2], dtype=tf.float32)

    @tf.function
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


def train_pi(state, action, r_sum, value, opt):
    with tf.GradientTape() as tape:
        mu = agent.model.pi_network(state)

        tfd = tfp.distributions
        dist = tfd.Normal(loc=mu, scale=agent.model.sigma)
        prob_of_action = dist.cdf(action)

        log_probability = tf.math.log(prob_of_action)

        advantage = r_sum - value

        gradients = tape.gradient(advantage * log_probability, agent.model.pi_network.trainable_variables)

        opt.apply_gradients(zip(-1 * gradients, agent.model.pi_network.trainable_variables))


def train_v(agent, state, mc_discounted, opt):
    with tf.GradientTape() as tape:
        value_estimate = agent.model.value_network(state)

        loss = tf.keras.losses.MSE(mc_discounted, value_estimate)  # + tf.reduce_sum(agent.model.value_network.losses)
        gradients = tape.gradient(loss, agent.model.value_network.trainable_variables)
        opt.apply_gradients(zip(gradients, agent.model.value_network.trainable_variables))

    return float(loss)


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
        "total_steps": 150,  # how many total steps we do
        "num_steps": 150,
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

    # buffer_size = 30000
    test_steps = 250
    epochs = 100
    saving_after = 10

    max_steps = 300

    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate)  # SGD(learning_rate, momentum=0.8)
    loss_function = tf.keras.losses.MSE

    ########################
    ## </Hyperparameters> ##
    ########################

    # keys for replay buffer -> what you will need for optimization
    # optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    # manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    # manager.initialize_aggregator(
    #     path=saving_path, saving_after=saving_after, aggregator_keys=["loss", "time_steps", "reward"]
    # )

    # initial testing:
    # print("Establishing baseline.")
    # manager.test(test_steps, test_episodes=3, do_print=True, render=True)

    print("Training the agent.")

    # get initial agent
    agent = manager.get_agent()
    agent.model.build((1, 8))
    env = manager.env_instance

    for e in range(epochs):
        print(">", end="")
        # data = manager.get_data()
        # manager.store_in_buffer(data)
        # sample_dict = manager.sample(1, from_buffer=False)

        states, actions, cumulative_mc_rewards = [], [], [],

        # get s0 and reshape it to make it batch size 1
        s = env.reset()
        for i in range(max_steps):
            s = np.array(s).reshape((1, 8))

            # obtain the mus for both action and sample the action from a gaussian afterwards.
            mu = agent.model.pi_network(s)
            a = tf.random.normal(shape=(2,), mean=mu, stddev=agent.model.sigma)

            # take the step in the env
            next_s, reward, done, _ = env.step(a.numpy().reshape(2))
            # next_s = np.array(next_s).reshape((1, 8))  # reshape immediately for the next iteration and for convenience.

            # save everything
            states.append(s)
            actions.append(a)
            # if list is empty we only need the reward as first value.
            # else, we are discounting cumulatively to obtain mc_rewards.
            cumulative_mc_rewards.append(reward if cumulative_mc_rewards == [] else cumulative_mc_rewards[-1] + (gamma ** i * reward))
            # new_states.append(next_s)

            s = next_s  # the posterior of today is tomorrows prior.

            if done:
                # if we reach a terminal state our estimate for the remaining trajectory is 0.
                cumulative_mc_rewards.append(0)
                break

        if not done:
            next_s = np.array(next_s).reshape((1, 8))  # bring it in shape :D
            # if we did not managed to reach a terminal state we will estimate the remaining trajectory.
            R = agent.model(next_s)['value_estimate']
            cumulative_mc_rewards.append(float(R))

        cumulative_mc_rewards = tf.constant(cumulative_mc_rewards, tf.float32)
        env.close()

        # create and batch tf datasets
        # data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        # def get_end(data):
        #     for i, d in enumerate(data):
        #         if d == 0:
        #             return i+1, True
        #     return len(data)-1, False

        # end, terminal = get_end(sample_dict['not_done'])

        # states = sample_dict['state'][:end]
        # actions = sample_dict['action'][:end]
        # new_states = sample_dict['state_new'][:end]
        # rewards = sample_dict['reward'][:end]
        # mc_rewards = [sum([gamma**j*r for j, r in enumerate(rewards[i:])]) for i in range(len(rewards))]
        # mc_rewards = sample_dict['monte_carlo'][:end]

        # print("Normal rewards: ", rewards)
        # print("MC rewards: ", mc_rewards)

        # if terminal: # TODO think about this
        #     end_state = states[end-1].reshape((1, 8))
        #     R = agent.model(end_state)['value_estimate']
        #     mc_rewards.append(gamma ** end * R)

        loss = 0
        
        # fix our value network to prevent inaccurate updates.
        stormtrooper = tf.keras.models.clone_model(agent.model.value_network)

        it = 1
        for s, a, mc_r in zip(states, actions, cumulative_mc_rewards):
            value_estimate = stormtrooper(s)

            train_pi(s, a, mc_r, value_estimate, optimizer)
            loss_v = train_v(agent, s, mc_r, optimizer)

            # old average * (n-1)/n + new value /n
            loss = loss * (it - 1) / it + loss_v / it
            it += 1

        # update with new weights
        new_weights = agent.model.get_weights()

        agent.set_weights(new_weights)
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        if e % 10 == 0:  # every 10 epochs do:
            if e % 20 == 0:  # render test.
                time_steps, rewards = manager.test(
                    test_steps,
                    test_episodes=10,
                    render=True,
                    evaluation_measure="time_and_reward"
                )
            else:  # or do not render the test and save model.
                time_steps, rewards = manager.test(test_steps, render=False, evaluation_measure="time_and_reward")
                # manager.save_model("./models", e, "LunarLander")

            # manager.update_aggregator(loss=loss, time_steps=time_steps, reward=rewards)
            print(f"\nepoch ::: {e}  loss ::: {loss}   reward ::: {np.sum(rewards)}   avg env steps ::: {np.mean(time_steps)}")

    print("\nDone.")
    print("Testing optimized agent.")

    manager.test(100, test_episodes=10, render=True, do_print=True)
