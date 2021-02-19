import os, logging
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import gridworlds
import gym
import ray
from really.utils import discount_cumsum
from ray.tune.registry import register_env



def env_creator(dict):
    return GridWorld()  # return an env instance

#
@ray.remote
class RunnerBox():
    """
    Runner Box handling interaction between an instance of the Agent and an instance of the environment.

    @args:
        agent: Agent Object
        model: callable Model object
        environment: string specifying gym environment or class of Custom (gym-like) environment
        runner_position: int, index an list of remote runners
        returns: list of strings specifying what is to be returned by the box
                supported are: 'value_estimate', 'log_prob', 'monte_carlo'

        kwargs:
                action_sampling_type: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', or 'continous_normal_diagonal'
                temperature: float, temperature for thomson sampling, defaults to 1
                epsilon: epsilon for epsilon greedy sampling, defaults to 0.95
                weights: weights of the model, not needed if input_shape is given
                model_kwargs: dict, optional, model specificatins requried for initialization
                gamma: float, discount factor for monte carlo return, defaults to 0.99
                $env_kwargs: dictionary, optional custom environment specifications
                input_shape: boolean, if model needs input shape for initial call, defaults to True
            """

    def __init__(self, agent, model, environment, runner_position, returns=[], **kwargs): #gamma=0.99 ,weights=None, num_actions=None, input_shape=None, type=None, temperature=1, epsilon=0.95, value_estimate=False):

        self.env = environment
        # if input shape is not set or nod needed, set to state shape fo model initialization
        if not('input_shape' in kwargs):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            kwargs['input_shape'] = state.shape

        self.agent = agent(model, **kwargs)
        self.agent_kwargs = kwargs
        self.runner_position = runner_position
        self.returns = returns

        self.return_log_prob = False
        self.return_value_estimate = False
        self.return_monte_carlo = False

        logging.basicConfig(filename=f'logging/box{self.runner_position}.log', level=logging.DEBUG)

        # initialize default data agg
        data_agg = {}
        data_agg['action'] = []
        data_agg['state'] = []
        data_agg['reward'] = []
        data_agg['state_new'] = []
        data_agg['terminal'] = []

        # initilize optional returns
        for key in self.returns:
            data_agg[key] = []

            if key == 'log_prob':
                self.return_log_prob = True
            elif key == 'value_estimate' and self.agent.value_estimate:
                self.return_value_estimate = True
            elif key == 'monte_carlo':
                self.return_monte_carlo = True
                if 'gamma' in kwargs.keys():
                    self.gamma = kwargs['gamma']
                else:
                    self.gamma = 0.99

        self.data_agg = data_agg

    #@ray.remote(num_returns=2)
    def run_n_steps(self, num_steps, max_env=None):
        import tensorflow as tf
        
        if max_env is not None: self.env.__num_steps = max_env
        state = self.env.reset()
        step = 0

        while step < num_steps:
            done = False
            new_state = np.expand_dims(self.env.reset(), axis=0)
            while not done:
                state = new_state
                agent_out = self.agent.act_experience(state, self.return_log_prob)

                # S
                self.data_agg['state'].append(state)
                # A
                action = agent_out['action']
                if tf.is_tensor(action):
                    action = action.numpy()
                new_state, reward, done, info = self.env.step(int(action))
                self.data_agg['action'].append(action)
                # R
                self.data_agg['reward'].append(reward)
                # S+1
                new_state = np.expand_dims(new_state, axis=0)
                self.data_agg['state_new'].append(new_state)
                # info on terminal state
                self.data_agg['terminal'].append(float(int(not(done))))

                # append optional in time values to data data
                if self.return_log_prob: self.data_agg['log_prob'].append(agent_out['log_probability'])
                if self.return_value_estimate: self.data_agg['value_estimate'].append(agent_out['value_estimate'])

                step += 1
                if step == num_steps: break

        if self.return_monte_carlo: self.data_agg['monte_carlo'] = discount_cumsum(self.data_agg['reward'], self.gamma)

        return self.data_agg, self.runner_position

    #@ray.remote(num_returns=2)
    def run_n_episodes(self, num_episodes, max_env = None):
        import tensorflow as tf
        if max_env is not None: self.env.__num_steps = max_env
        state = self.env.reset()
        for e in range(num_episodes):
            done = False
            new_state = np.expand_dims(self.env.reset(), axis=0)
            while not done:
                state = new_state
                agent_out = self.agent.act_experience(state, self.return_log_prob)
                # S
                self.data_agg['state'].append(state)
                # A
                action = agent_out['action']
                if tf.is_tensor(action):
                    action = action.numpy()
                # A
                new_state, reward, done, info = self.env.step(int(action))
                self.data_agg['action'].append(action)
                # R
                self.data_agg['reward'].append(reward)
                # S+1
                new_state = np.expand_dims(new_state, axis=0)
                self.data_agg['state_new'].append(new_state)
                # info on terminal state
                self.data_agg['terminal'].append(int(not(done)))

                # append optional in time values to data data
                if self.return_log_prob: self.data_agg['log_prob'].append(agent_out['log_probability'])
                if self.return_value_estimate: self.data_agg['value_estimate'].append(agent_out['value_estimate'])

        if self.return_monte_carlo: self.data_agg['monte_carlo'] = discount_cumsum(self.data_agg['reward'], self.gamma)

        return self.data_agg, self.runner_position

    def get_agent_kwargs(self):
        agent_kwargs = self.agent_kwargs
        return agent_kwargs

    def env_creator(self, object, **kwargs):
        return object(**kwargs)
