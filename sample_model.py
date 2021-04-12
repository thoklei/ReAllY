import gym
import numpy as np
import ray
import os
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)
from really.utils import discount_cumsum

import tensorflow as tf
from tensorflow.keras import Model

class A2C(Model):
    def __init__(self, layers, action_dim):
        super(A2C, self).__init__()
        self.mu_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_mu_{i}'
                ) for i, num_units in enumerate(layers)]

        self.readout_mu = tf.keras.layers.Dense(units=action_dim,
                                                activation=None,
                                                name='Policy_mu_readout'
                                                )

        self.sigma_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_sigma_{i}'
                ) for i, num_units in enumerate(layers)]
                
        self.readout_sigma = tf.keras.layers.Dense(units=action_dim,
                                                   activation=None,
                                                   name='Policy_sigma_readout'
                                                   )

        self.value_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Value_layer_{i}'
                ) for i, num_units in enumerate(layers)]
                
        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation=None,
                                                   name='Value_readout'
                                                   )

    @tf.function
    def call(self, input_state):
        output = {}
        mu_pred = input_state
        sigma_pred = input_state
        value_pred = input_state
        for layer in self.mu_layer:
            mu_pred = layer(mu_pred)
        for layer in self.sigma_layer:
            sigma_pred = layer(sigma_pred)
        for layer in self.value_layer:
            value_pred = layer(value_pred)

        # Actor
        output["mu"] = tf.squeeze(self.readout_mu(mu_pred))
        output["sigma"] = tf.squeeze(tf.abs(self.readout_sigma(sigma_pred)))
        # Critic
        output["value_estimate"] = tf.squeeze(self.readout_value(value_pred))
        return output

