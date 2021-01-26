
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

    def __init__(agent, environment_name, returns=['reward'], gamma=None):

        self.agent = agent()
        self.environment = gym.make(environment_name)
        self.returns = returns
        self.gamma = gamma


    def run(num_steps):

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
            elif key == 'value_estimate' and agent.value_estimate:
                return_value_estimate = True
            elif key == 'monte_carlo' and gamma is not None:
                return_monte_carlo = True
            else:
                print(f"unsupported key: {key}")

        for t in range(num_steps):
            self.env.render()
            state = np.expand_dims(state, axis=0)
            agent_out = agent.act(state, return_log_prob)
            state, reward, done, info = self.env.step(agent_out['action'])

            # append optional in time values to data data
            data_agg['action'].append(agent_out['action'])
            data_agg['state'].append(state)

            if return_log_prob: data_agg['log_prob'].append(agent_out['log_probability'])
            if return_reward: data_agg['reward'].append(reward)
            if return_value_estimate: data_agg['value_estimate'].append(agent_out['value_estimate'])

            if done:
                break

        if return_monte_carlo: data_agg['monte_carlo'] = discount_cumsum(data_agg['reward'], gamma)


        return data_agg
