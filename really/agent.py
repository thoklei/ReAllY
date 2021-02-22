import os, logging
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
            model: tf.keras.Model (or simply callable bit than some functionalities are not supported) returning dictionary with the possible keys: 'q_values' or 'mus' and 'sigmas' for continuius cases, optional 'value_estimate', containing tensors
            weights: None or weights of the model
            action_sampling_type: string, supported are 'thompson', 'epsilon_greedy' and 'continous_normal_diagonal'
            epsilon: float for epsilon greedy sampling
            temperature: float for thompson sampling
            value_estimate: boolean if agent returns value estimate
            model_kwargs: dict, optional model initialization specifications
    """

    def __init__(self, model, weights=None, action_sampling_type='thompson', temperature=1, epsilon=0.95, value_estimate=False, input_shape=None, model_kwargs={}):
        super(Agent, self).__init__

        self.model_kwargs = model_kwargs
        self.model = model(**self.model_kwargs)

        self.weights = weights

        self.action_sampling_type = action_sampling_type
        self.temperature = temperature
        self.epsilon = epsilon
        self.value_estimate = value_estimate
        logging.basicConfig(filename=f'logging/agent.log', level=logging.DEBUG)
        logging.warning(f'input shape {input_shape}')
        # if weights given iitialize random weights, else set weights
        self.initialize_weights(self.model, input_shape)

        if weights is not None:
            self.model.set_weights(weights)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def initialize_weights(self, model, input_shape):
        if not(input_shape):
            return model.get_weights()
        if hasattr(model, 'tensorflow'):
            assert input_shape!=None, 'You have a tensorflow model with no input shape specified for weight initialization. \n Specify input_shape in "model_kwargs" or specify as False if not needed'
        dummy = tf.zeros(input_shape)
        model(dummy)
        weights = model.get_weights()

        return weights

    # agent readout handler
    def act_experience(self, state, return_log_prob=False):
        output = {}
        # creating network dict
        network_out = self.model(state)

        if self.action_sampling_type == 'epsilon_greedy':
            logits = network_out['q_values']
            if tf.is_tensor(logits):
                logits= logits.numpy()
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


        elif self.action_sampling_type == 'thompson':
            # q values
            logits = network_out['q_values']
            if tf.is_tensor(logits):
                logits= logits.numpy()
            logits  = logits/self.temperature
            action = tf.squeeze(tf.random.categorical(logits, 1))
            output['action'] = action
            if return_log_prob: output['log_probability'] = np.log([probs[i][a] for i,a in zip(range(probs.shape[0]), action)])

        elif self.action_sampling_type == 'continous_normal_diagonal':

            mus, sigmas = network_out['mu'].numpy(), network_out['sigma'].numpy()
            action = norm.rvs(mus, sigmas)
            output['action'] = action

            if return_log_prob: output['log_probability'] = np.sum(norm.logpdf(action, mus, sigmas))
        else:
            #logging.warning(f'unsupported sampling method {self.actin_sampling_type}')
            raise NotImplemented
        # pass on value estimate if there
        if self.value_estimate :
            output['value_estimate'] = network_out['value_estimate']

        return output

    def act(self, state):
        net_out = self.act_experience(state)
        return net_out['action']

    def max_q(self, x):
        # computes the maximum q-value along each batch dimension
        model_out = self.model(x)
        x = tf.reduce_max(model_out['q_values'], axis=-1)
        return x

    def q_val(self, x, actions):
        model_out = self.model(x)
        q_values = model_out['q_values']
        x = tf.gather(q_values, actions, batch_dims=1)
        return x
