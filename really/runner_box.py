import os
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import gym
import ray
from really.utils import discount_cumsum

@ray.remote
class RunnerBox():
    """
    Runner Box handling interaction between an instance of the Agent and an instance of the environment.

    @args:
        agent: Agent Object
        model: ANN Model object
        environment_name: string specifying gym environment
        runner_position: int, index an list of remote runners
        returns: list of strings specifying what is to be returned by the box
                supported are: 'value_estimate', 'log_prob', 'monte_carlo'

        kwargs:
                action_sampling_type: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', or 'continous_normal_diagonal'
                temperature: float, temperature for thomson sampling, defaults to 1
                epsilon: epsilon for epsilon greedy sampling, defaults to 0.95
                weights: weights of the model, not needed if input_shape is given
                needs_output_shape = True, boolean specifying if the number of actions needs to be passed on to the model for first initialization
                gamma: float, discount factor for monte carlo return, defaults to 0.99
            """

    def __init__(self, agent, model, environment_name, runner_position, returns=[], **kwargs): #gamma=0.99 ,weights=None, num_actions=None, input_shape=None, type=None, temperature=1, epsilon=0.95, value_estimate=False):

        self.env = gym.make(environment_name)
        self.agent = agent(model, **kwargs)
        self.runner_position = runner_position
        self.returns = returns

        self.return_log_prob = False
        self.return_value_estimate = False
        self.return_monte_carlo = False

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

    @ray.remote(num_returns=2)
    def run_n_steps(self, num_steps, max_env=None):

        if max_env is not None: self.env.__num_steps = max_env
        state = self.env.reset()
        step = 0

        while step < num_steps:
            done = False
            new_state = np.expand_dims(self.env.reset(), axis=0)
            while not done:
                state = new_state
                agent_out = self.agent.act(state, self.return_log_prob)

                # S
                self.data_agg['state'].append(state)
                # A
                new_state, reward, done, info = self.env.step(agent_out['action'].numpy())
                self.data_agg['action'].append(agent_out['action'].numpy())
                # R
                self.data_agg['reward'].append(reward)
                # S+1
                new_state = np.expand_dims(new_state, axis=0)
                self.data_agg['state_new'].append(new_state)
                # info on terminal state
                self.data_agg['terminal'].append(int(done))

                # append optional in time values to data data
                if self.return_log_prob: self.data_agg['log_prob'].append(agent_out['log_probability'])
                if self.return_value_estimate: self.data_agg['value_estimate'].append(agent_out['value_estimate'])

                step += 1
                if step == num_steps: break

        if self.return_monte_carlo: self.data_agg['monte_carlo'] = discount_cumsum(self.data_agg['reward'], self.gamma)

        return self.data_agg, self.runner_position

    def run_n_episodes(self, num_episodes, max_env = None):
        if max_env is not None: self.env.__num_steps = max_env
        state = self.env.reset()
        for e in range(num_episodes):
            done = False
            new_state = np.expand_dims(self.env.reset(), axis=0)
            while not done:
                state = new_state
                agent_out = self.agent.act(state, self.return_log_prob)

                # S
                self.data_agg['state'].append(state)
                # A
                new_state, reward, done, info = self.env.step(agent_out['action'].numpy())
                self.data_agg['action'].append(agent_out['action'].numpy())
                # R
                self.data_agg['reward'].append(reward)
                # S+1
                new_state = np.expand_dims(new_state, axis=0)
                self.data_agg['state_new'].append(new_state)
                # info on terminal state
                self.data_agg['terminal'].append(int(done))

                # append optional in time values to data data
                if self.return_log_prob: self.data_agg['log_prob'].append(agent_out['log_probability'])
                if self.return_value_estimate: self.data_agg['value_estimate'].append(agent_out['value_estimate'])

        if self.return_monte_carlo: self.data_agg['monte_carlo'] = discount_cumsum(self.data_agg['reward'], self.gamma)

        return self.data_agg, self.runner_position
