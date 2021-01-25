
import tensorflow as tf
import numpy as np
import random


# agent readout handler
def readout_handler(model, state)
    type = 'thompson'
    epsilon = 0.05
    temperature = 1

    if self.type == 'epsilon_greedy':
        q_values = model(state).numpy()
        if random.random > epsilon
            action = np.argmax(q_values)
        else:
            action = random.randrange(q_values.shape[-1])

    if self.type == 'thompson':
        # q values
        q_values = model(state)/temperature
        q_ps = tf.nn.softmax(q_values)
        # get index of maximum
        action = np.argmax(q_values.numpy())

    if self.type == 'continous_normal_diagonal':
        mus, sigmas = model(state).numpy()
        action = np.random.normal(mus, sigmas, size=1)

    return action




class Agent():
    """
    Agent wrapper:
        @args
            model: tf.keras.Model (callable)
            weights: corresponding weights
            input_shape: size of input
            readout_handler: function taking (model, state) and returning action
    """

    def __init__(self, model, readout_handler ,weights=None, input_shape=None):
        super(Agent, self).__init__()
        self.model = model()

        # do we need thaa?
        self.weights = weights
        self.input_shape = input_shape
        self.readout_handler = readout_handler

        # if now weights given iitialize random weights, else set weights
        if weights is not None:
            self.model.set_weights(weights)
        else:
            random_weights = self.initialize_weights(self.model, input_shape)
            #m do we need that?


    def set_weights(self, weights):
        self.model.set_weights(weights)


    def initialize_weights(self, model, input_shape):
        assert input_shape!=None, 'no input shape specified for weight initialization'
        dummy = tf.zeros(input_shape)
        model(dummy)
        weights = model.get_weights()

        return weights


    def act(self, state):
        action = self.readout_handler(state)
        return action







# env
# small box
# big box
