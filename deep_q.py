import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Model
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from really.sample_manager import SampleManager


class DQN(Model):
    def __init__(self, actions):
        super(DQN, self).__init__()
        self.in_layer = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.hidden_1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.hidden_2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(actions)

    def call(self, x):
        output = {}
        x = self.in_layer(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        output['output'] = x
        return output


def train_step(agent, states, actions, rewards, states_new, terminal):

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
    env = gym.make('CartPole-v0')
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    input_shape = state.shape
    num_actions = env.action_space.n
    model = DQN(num_actions)
    model(tf.ones(input_shape))
    weights = model.get_weights()

    kwargs = {
        'model' : DQN,
        'environment_name' :'CartPole-v0',
        'num_parallel' :10,
        'total_steps' :10000,
        'action_sampling_type' :'thompson',
        'output_shape' : num_actions,
        'weights' : weights,
        'num_episodes': 20,

    }

    manager = SampleManager(**kwargs)

    saving_path = os.getcwd()+'/progress'
    buffer_size = 50000
    test_steps = 1000
    epochs = 100
    sample_size = int(0.05*buffer_size)
    optim_batch_size = 64
    gamma = 0.95
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.MSE
    saving_after = 2

    # keys needed for deep q
    optim_keys = ['state', 'action', 'reward', 'state_new', 'terminal']

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)
    agent = manager.get_agent()

    # initilize progress aggregator
    manager.initialize_aggregator(path=saving_path, saving_after=saving_after, aggregator_keys=['loss', 'time_steps'])

    # initial testing:
    print('test before training: ')
    manager.test(test_steps, do_print=True)

    for e in range(epochs):
        # anneal sampling temperature
        temperature = 1/(1+(e/20))
        manager.set_temperature(temperature)

        print('collecting experience..')
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        experience_dict = manager.sample_dictionary_of_datasets(sample_size)
        # batch datasets
        for k in experience_dict:
            experience_dict[k] = experience_dict[k].batch(optim_batch_size)

        print('optimizing...')
        losses = []
        #for t in range(train_steps):
        for states, actions, rewards, states_new, terminals in zip(*[experience_dict[k] for k in optim_keys]):
            rewards = np.expand_dims(rewards, axis=-1)
            terminals = np.expand_dims(terminals, axis=-1)
            loss = train_step(agent, states, actions, rewards, states_new, terminals)
            losses.append(np.mean(loss, axis=0))
        # set new weights, get optimized agent
        manager.set_agent(agent.model.get_weights())
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_agg(loss=losses, time_steps=time_steps)
        print(f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses])}   avg env steps ::: {np.mean(time_steps)}"   )

    print('done')
    print('testing optimized agent')
    manager.set_temperature(1/epochs)
    manager.test(test_steps, render=True)
