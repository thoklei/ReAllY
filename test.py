# test script for debugging

import os
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import gym

from really.sample_manager import SampleManager



class MyModel(tf.keras.Model):

    def __init__(self, output_units=2):

        super(MyModel, self).__init__()
        self.layer = tf.keras.layers.Dense(output_units)
        self.layer2 = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        x = self.layer(x_in)
        v = self.layer2(x)
        output['output'] = x
        output['value_estimate'] = v
        return output

class ModelContunous(tf.keras.Model):

    def __init__(self, output_units=2):

        super(ModelContunous,self).__init__()

        self.layer_mu = tf.keras.layers.Dense(output_units)
        self.layer_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        mus = self.layer_mu(x_in)
        sigmas = tf.exp(self.layer_sigma(x_in))
        v = self.layer_v(x_in)
        output['mu'] = mus
        output['sigma'] = sigmas
        output['value_estimate'] = v

        return output


if __name__== "__main__":


    env = gym.make('CartPole-v0')
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    input_shape = state.shape
    num_actions = env.action_space.n

    kwargs = {
        'model' : MyModel,
        'environment_name' :'CartPole-v0',
        'num_parallel' :10,
        'total_steps' :100,
        'returns' :['value_estimate', 'monte_carlo'],
        'action_sampling_type' :'thompson',
        'num_actions' :True,
        'input_shape' : input_shape,
        'remote_min_returns' :3

    }
    manager = SampleManager(**kwargs)

    data_dict = manager.create_dictionary_of_datasets()
    print(data_dict.keys())

    actions = data_dict['action'].batch(32)
