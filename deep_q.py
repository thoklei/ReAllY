import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Model
import gym
import random
import numpy as np
import ray
import matplotlib.pyplot as plt
from really import SampleManager
from really.utils import dict_to_dict_of_datasets


class DQN(Model):
    def __init__(self, actions):
        super(DQN, self).__init__()
        self.in_layer = tf.keras.layers.Dense(16, activation=tf.nn.tanh)
        self.hidden_1 = tf.keras.layers.Dense(16, activation=tf.nn.tanh)
        self.hidden_2 = tf.keras.layers.Dense(16, activation=tf.nn.tanh)
        self.out = tf.keras.layers.Dense(actions, use_bias=False)

    def call(self, x):
        output = {}
        x = self.in_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        output['q_values'] = x
        return output


def train_step(optimizer, agent, states, actions, rewards, states_new, terminal):

    max_qs = agent.max_q(states_new)
    targets = rewards + terminal * gamma * max_qs
    # make sure targets are detached
    targets = targets.numpy()
    with tf.GradientTape() as tape:
        q_vals = agent.q_val(states, actions)
        loss = loss_function(targets, q_vals)
    gradients = tape.gradient(loss, agent.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

    return loss


if __name__== "__main__":
    ray.init(log_to_driver=False)
    env = gym.make('LunarLander-v2')
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    input_shape = state.shape
    num_actions = env.action_space.n
    model = DQN(num_actions)
    model(tf.ones(input_shape))
    weights = model.get_weights()

    model_kwargs = {
        'actions' : num_actions
    }

    epsilon = 1
    epsilo_decay_factor = 0.99
    kwargs = {
        'model' : DQN,
        'model_kwargs': model_kwargs,
        'environment' :'LunarLander-v2',
        'num_parallel' :8,
        'total_steps' : 400,
        'action_sampling_type' :'epsilon_greedy',
        'weights' : weights,
        'num_episodes': 20,
        'epsilon' : 0.85

    }


    manager = SampleManager(**kwargs)

    saving_path = os.getcwd()+'/progress_deep_q'
    buffer_size = 30000
    test_steps = 100
    epochs = 1000
    sample_size = 4000
    optim_batch_size = 64
    gamma = 0.85
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    loss_function = tf.keras.losses.MSE
    saving_after = 20

    # keys needed for deep q
    optim_keys = ['state', 'action', 'reward', 'state_new', 'terminal']

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)
    agent = manager.get_agent()

    # initilize progress aggregator
    manager.initialize_aggregator(path=saving_path, saving_after=saving_after, aggregator_keys=['loss', 'time_steps', 'rewards'])
    # fill buffer
    data = manager.get_data(total_steps=buffer_size)
    #print(data['terminal'])
    manager.store_in_buffer(data)
    # initial testing:
    print('test before training: ')
    manager.test(test_steps, do_print=True, evaluation_measure='time_and_reward')

    for e in range(epochs):
        # anneal sampling temperature
        epsilon = max(epsilon * epsilo_decay_factor,0.05)
        manager.set_epsilon(epsilon)

        print('collecting experience..')
        data = manager.get_data()
        #print(data['terminal'])
        manager.store_in_buffer(data)
        # sample from buffer
        sample_dict = manager.sample(sample_size)
        # create dict of tf datasets
        dataset_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print('optimizing...')
        losses = []
        #for t in range(train_steps):
        for states, actions, rewards, states_new, terminals in zip(*[dataset_dict[k] for k in optim_keys]):
            rewards = np.expand_dims(rewards, axis=-1)
            terminals = np.expand_dims(terminals, axis=-1)
            loss = train_step(optimizer, agent, states, actions, rewards, states_new, terminals)
            #print(f'loss: {loss}')
            losses.append(np.mean(loss, axis=0))
        # set new weights, get optimized agent
        manager.set_agent(agent.model.get_weights())
        agent = manager.get_agent()
        # update aggregator
        time_steps, reward_agg = manager.test(test_steps, evaluation_measure='time_and_reward')
        manager.update_agg(loss=losses, time_steps=time_steps, rewards=reward_agg)
        print(f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses])}   avg env steps ::: {np.mean(time_steps)}   avg agg reward ::: {np.mean(reward_agg)}")
        if e%saving_after==0:
            manager.save_model(saving_path, e)
            manager.test(test_steps, test_episodes=10, render=True, evaluation_measure='time_and_reward')

    print('done')
    print('testing optimized agent')
    manager.test(test_steps, render=True)
