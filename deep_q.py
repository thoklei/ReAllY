import tensorflow as tf
from tensorflow.keras.models import Model
import gym
import random
import numpy as np
from agent import Agent
from runner_box import RunnerBox

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


def train_step(states, actions, rewards, states_new):

    with tf.GradientTape() as tape:
        max_qs = agent.max_q(states_new)
        targets = rewards + gamma * max_qs
        q_vals = agent.q_val(states, actions)
        loss = loss_function(targets, q_vals)
        gradients = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, agent.trainable_variables))
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

    #agent = Agent(DQN, num_actions=num_actions, input_shape=input_shape, type='continous_normal_diagonal', value_estimate=True)

    box = RunnerBox(Agent, DQN, 'CartPole-v0', returns=[], type='thompson', num_actions=num_actions, input_shape=input_shape)

    data = box.run(100)
    print(data.keys())
    #print(data)
