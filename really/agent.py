import os
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import random
from scipy.stats import norm, lognorm


class Agent():
    """
    Agent wrapper:
        @args
            model: tf.keras.Model (callable) returning dictionary with the possible keys: 'output' or 'mus' and 'sigmas' for continuius cases, optional 'value_estimate', containing tensors
            num_actions: int or None, specifiying the output shape of the model
            weights: None or weights of the model
            input_shape: None or size of input, only needed when no weights are given
            action_sampling_type: string, supported are 'thompson', 'epsilon_greedy' and 'continous_normal_diagonal'
            epsilon: float for epsilon greedy sampling
            temperature: float for thompson sampling
            value_estimate: boolean if agent returns value estimate
    """

    def __init__(self, model, num_actions=None, weights=None, input_shape=None, action_sampling_type='thompson', temperature=1, epsilon=0.95, value_estimate=False):
        super(Agent, self).__init__

        if num_actions is not None:
            self.model = model(num_actions)
        else: self.model = model()

        # do we need thaa?
        self.num_actions = num_actions
        self.weights = weights
        self.input_shape = input_shape

        self.action_sampling_type = action_sampling_type
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

        if self.action_sampling_type == 'epsilon_greedy':
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


        if self.action_sampling_type == 'thompson':
            # q values
            logits = network_out['output'].numpy()
            logits  = logits/self.temperature
            action = tf.squeeze(tf.random.categorical(logits, 1))
            output['action'] = action
            if return_log_prob: output['log_probability'] = np.log([probs[i][a] for i,a in zip(range(probs.shape[0]), action)])

        if self.action_sampling_type == 'continous_normal_diagonal':

            mus, sigmas = network_out['mu'].numpy(), network_out['sigma'].numpy()
            action = norm.rvs(mus, sigmas)
            output['action'] = action

            if return_log_prob: output['log_probability'] = np.sum(norm.logpdf(action, mus, sigmas))

        # pass on value estimate if there
        if self.value_estimate :
            output['value_estimate'] = network_out['value_estimate']

        return output

    def max_q(self, x):
        #print('max q x', x.shape)
        # computes the maximum q-value along each batch dimension
        model_out = self.model(x)
        x = tf.reduce_max(model_out['output'], axis=-1)
        return x

    def q_val(self, x, actions):
        print('q_val x', x.shape)
        print(' q val action', actions)
        # for each action return the q_value
        model_out = self.model(x)
        print('out', model_out['output'].shape)
        x = tf.gather(model_out['output'], actions, batch_dims=0)
        return x






# env
# small box
# big box
