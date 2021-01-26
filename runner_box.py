
import tensorflow as tf
import numpy as np
import gym

from agent import Agent
from utils import discount_cumsum

class RunnerBox():
    """
    Runner Box handling interaction between an instance of the Agent and an instance of the environment.

    @args:
        agent: Agent Object
        environment_name: string specifying gym environment
        returns: list of strings specifying what is to be returned by the box
                supported are: 'reward', 'value_estimate', 'log_prob', 'monte_carlo'
        gamma: float, discount factor for monte carlo return
            """

    def __init__(self, agent, model, environment_name, returns=['reward'], gamma=0.99 ,weights=None, input_shape=None, type=None, temperature=1, epsilon=0.95, value_estimate=False):

        self.agent = agent(model ,weights, input_shape, type, temperature, epsilon, value_estimate)
        self.env = gym.make(environment_name)
        self.returns = returns
        self.gamma = gamma


    def run(self, num_steps):

        return_log_prob = False
        return_value_estimate = False
        return_monte_carlo = False
        return_reward = False

        state = self.env.reset()
        data_agg = {}

        # initialize default data data data
        data_agg['action'] = []
        data_agg['state'] = []

        # initilize optional returns
        for key in self.returns:
            data_agg[key] = []

            if key == 'log_prob':
                return_log_prob = True
            elif key == 'value_estimate' and self.agent.value_estimate:
                return_value_estimate = True
            elif key == 'monte_carlo' and self.gamma is not None:
                return_monte_carlo = True
            elif key == 'reward':
                return_reward = True
            else:
                print(f"unsupported key: {key}")

        for t in range(num_steps):
            self.env.render()
            state = np.expand_dims(state, axis=0)
            agent_out = self.agent.act(state, return_log_prob)

            # if continous pass on array as action
            if self.agent.type == 'continous_normal_diagonal':
                state, reward, done, info = self.env.step(agent_out['action'])
                data_agg['action'].append(agent_out['action'])

            else:
                state, reward, done, info = self.env.step(*agent_out['action'])
                data_agg['action'].append(*agent_out['action'])
            data_agg['state'].append(state)

            # append optional in time values to data data
            if return_log_prob: data_agg['log_prob'].append(agent_out['log_probability'])
            if return_reward: data_agg['reward'].append(reward)
            if return_value_estimate: data_agg['value_estimate'].append(agent_out['value_estimate'])

            if done:
                break

        if return_monte_carlo: data_agg['monte_carlo'] = discount_cumsum(data_agg['reward'], self.gamma)


        return data_agg
