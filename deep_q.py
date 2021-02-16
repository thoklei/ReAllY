import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model
import gym
import random
import numpy as np

from really.sample_manager import SampleManager
from really.buffer import Replay_buffer

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


def test(test_steps, agent, env, test_episodes = 100, render = False):

    for _ in range(test_episodes):
        state_new = np.expand_dims(env.reset(), axis=0)
        time_steps = []

        for t in range(test_steps):
            if render: env.render()
            state = state_new
            action = agent.act(state).numpy()
            state_new, reward, done, info = env.step(action)
            state_new = np.expand_dims(state_new, axis=0)

            if done:
                time_steps.append(t)
                break
            if t == test_steps:
                time_steps.append(t)
                break

    env.close()
    avg = np.mean(time_steps)
    print(f"Episodes finished after a mean of {avg} timesteps")
    return avg


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

    buffer_size = 50000
    test_steps = 1000
    epochs = 100
    sample_size = int(0.05*buffer_size)
    optim_batch_size = 64
    gamma = 0.95
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.MSE
    moving_average = 0
    moving_average_steps = 0
    moving_average_factor = 0.95
    losses_e = []
    env_staps = []

    # keys needed for deep q
    optim_keys = ['state', 'action', 'reward', 'state_new', 'terminal']

    buffer = Replay_buffer(buffer_size, optim_keys)
    agent = manager.get_agent()

    # initial testing:
    print('test before training: ')
    test(test_steps, agent, env)


    for e in range(epochs):
        # anneal sampling temperature
        temperature = 1/(1+(e/20))
        manager.set_temperature(temperature)

        print('collecting experience..')
        data = manager.get_data()
        buffer.put(data)

        # sample data to optimize on from buffer
        experience_dict = buffer.sample_dictionary_of_datasets(sample_size)
        # batch datasets
        for k in experience_dict:
            experience_dict[k] = experience_dict[k].batch(optim_batch_size)

        print('optimizing...')

        #for t in range(train_steps):
        for states, actions, rewards, states_new, terminals in zip(*[experience_dict[k] for k in optim_keys]):
            losses = []
            rewards = np.expand_dims(rewards, axis=-1)
            terminals = np.expand_dims(terminals, axis=-1)
            #print(f'states {states.shape}')
            #print(f'rewards {rewards.shape}')
            #print(f'states n {states_new.shape}')
            #print(f'terminal {terminals.shape}')
            loss = train_step(agent, states, actions, rewards, states_new, terminals)
            losses.append(loss)

        # set new weights, get optimized agent
        manager.set_agent(agent.model.get_weights())
        agent = manager.get_agent()

        moving_average = moving_average_factor * np.mean(losses) + (1-moving_average_factor) * moving_average
        losses_e.append(moving_average)
        avg_t = test(test_steps, agent, env)
        moving_average_steps = moving_average_factor * avg_t + (1-moving_average_factor) * moving_average_steps
        env_staps.append(moving_average_steps)
        print(f"epoch ::: {e}  loss ::: {moving_average}   avg env steps ::: {moving_average_steps}"   )

    print('done')
    plt.plot(losses_e)
    plt.plot(env_staps)
    plt.show()
    print('testing optimized agent')
    manager.set_temperature(1/epochs)
    test(test_steps, agent, env, render=True)
