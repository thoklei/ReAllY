
import tensorflow as tf
import numpy as np
import random
from scipy.stats import norm, lognorm


class Agent():
    """
    Agent wrapper:
        @args
            model: tf.keras.Model (callable) returning dictionary with the possible keys: 'output' or 'mus' and 'sigmas' for continuius cases, optional 'value_estimate', containing tensors
            weights: corresponding weights
            input_shape: size of input, only needed when no weights are given
            type: string, supported are 'thompson', 'epsilon_greedy' and 'continous_normal_diagonal'
            epsilon: float for epsilon greedy sampling
            temperature: float for thompson sampling
            value_estimate: boolean if agent returns value estimate
    """

    def __init__(self, model ,weights=None, input_shape=None, type=None, temperature=1, epsilon=0.95, value_estimate=False):
        super(Agent, self).__init__()
        self.model = model()

        # do we need thaa?
        self.weights = weights
        self.input_shape = input_shape
        self.type = type
        self.temperature = temperature
        self.epsilon = epsilon
        self.value_estimate = value_estimate

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

    # agent readout handler
    def act(self, state, return_log_prob=False):
        output = {}
        # creating network dict
        network_out = self.model(state)
        print('net_out', network_out)

        if self.type == 'epsilon_greedy':
            logits = network_out['output'].numpy()
            if random.random() > self.epsilon:
                action = np.argmax(logits, axis=-1)
                # log prob of epsilon
                if return_log_prob: output['log_probability'] = np.asarray([np.log(self.epsilon)]*logits.shape[0])

            else:
                # log prob of 1-epsilon
                action = [random.randrange(logits.shape[-1]) for _ in range(logits.shape[0])]
                action = np.asarray(action)
                if return_log_prob: output['log_probability'] = np.asarray([np.log(1-self.epsilon)]*logits.shape[0])
            output['action'] = action


        if self.type == 'thompson':
            # q values
            logits = network_out['output'].numpy()
            logits  = logits/self.temperature
            probs = tf.nn.softmax(logits, axis=-1)
            # get index of maximum
            action = np.argmax(probs.numpy(), axis=-1)
            output['action'] = action
            if return_log_prob: output['log_probability'] = np.log([probs[i][a] for i,a in zip(range(probs.shape[0]), action)])

        if self.type == 'continous_normal_diagonal':

            mus, sigmas = network_out['mu'].numpy(), network_out['sigma'].numpy()
            action = norm.rvs(mus, sigmas)
            output['action'] = action

            if return_log_prob: output['log_probability'] = np.sum(norm.logpdf(action, mus, sigmas))

        # pass on value estimate if there
        if self.value_estimate :
            output['value_estimate'] = network_out['value_estimate']

        return output









# env
# small box
# big box
