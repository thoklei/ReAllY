import os
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ray
import tensorflow as tf
import gym
import numpy as np
from really.agent import Agent
from really.runner_box import RunnerBox
from really.buffer import Replay_buffer
from really.agg import Smoothing_aggregator


class SampleManager():

    """
    model: model Object
    environment_name: string specifying gym environment
    num_parallel: int, number of how many agents to run in parall
    total_steps: int, size of the total steps collected
    returns: list of strings specifying what is to be returned by the box
            supported are: 'value_estimate', 'log_prob', 'monte_carlo'
    actin_sampling_type: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', or 'continous_normal_diagonal'
    kwargs:
        num_episodes: specifies the total number of episodes to run on the environment for each runner, defaults to 1
        num_steps: specifies the total number of steps to run on the environment for each runner
        gamma: float, discount factor for monte carlo return, defaults to 0.99
        temperature: float, temperature for thomson sampling, defaults to 1
        epsilon: epsilon for epsilon greedy sampling, defaults to 0.95
        weights: weights of the model, not needed if input_shape is given
        needs_output_shape: True, boolean specifying if the number of actions needs to be passed on to the model for first initialization
        remote_min_returns: int, minimum number of remote runner results to wait for, defaults to 10% of num_parallel
        remote_time_out: float, maximum amount of time (in seconds) to wait on the remote runner results, defaults to None
    """

    def __init__(self, model, environment_name, num_parallel, total_steps, returns=[], **kwargs):


        self.model = model
        self.environment_name = environment_name
        self.num_parallel = num_parallel
        self.total_steps = total_steps
        self.returns = returns
        self.kwargs = kwargs
        self.buffer = None

        # initilize empty dete aggregator
        self.data = {}
        self.data['action'] = []
        self.data['state'] = []
        self.data['reward'] = []
        self.data['state_new'] = []
        self.data['terminal'] = []

        ## TODO some checkups

        assert self.num_parallel > 0, 'num_parallel hast to be greater than 0!'

        # check action sampling type
        if 'action_sampling_type' in kwargs.keys():
            type = kwargs['action_sampling_type']
            if type not in ['thompson', 'epsilon_greedy', 'continous_normal_diagonal']:
                print(f'unsupported sampling type: {type}. assuming thompson sampling instead.')
                self.kwargs['action_sampling_type'] = 'thompson'


        # chck return specifications
        for r in returns:
            if r not in ['log_prob', 'monte_carlo', 'value_estimate']:
                print(f'unsuppoerted return key: {r}')
                if r == 'value_estimate':
                    self.kwargs['value_estimate'] = True
            else: self.data[r] = []

        ## check if model can be initialized

        ## info on default values

        ## check for runner sampling method:
        # error if both are specified
        self.run_episodes = True
        self.runner_steps = 1
        if 'num_episodes' in kwargs.keys():
            self.runner_steps = kwargs['num_episodes']
            if 'num_steps' in kwargs.keys():
                raise 'Both episode mode and step mode for runner sampling are specified. Please only specify one.'
            self.kwargs.pop('num_episodes')
        elif 'num_steps' in kwargs.keys():
            self.runner_steps = kwargs['num_steps']
            self.run_episodes = False
            self.kwargs.pop('num_steps')

        # check for remote process specifications
        if 'remote_min_returns' in kwargs.keys():
            self.remote_min_returns = kwargs['remote_min_returns']
            self.kwargs.pop('remote_min_returns')
        else:
            # defaults to 10% of remote runners, but minimum 1
            self.remote_min_returns = max([int(0.1 * self.num_parallel),1])

        if 'remote_time_out' in kwargs.keys():
            self.remote_time_out = kwargs['remote_time_out']
            self.kwargs.pop('remote_time_out')
        else:
            # defaults to None, i.e. wait for remote_min_returns to be returned irrespective of time
            self.remote_time_out = None


    def get_data(self, do_print=False):
        ray.shutdown()
        # surrpressing warnings
        ray.init(log_to_driver=False)

        not_done = True
        # create list of runnor boxes
        runner_boxes = [RunnerBox.remote(Agent, self.model, self.environment_name, runner_position=i, returns=self.returns, **self.kwargs) for i in range(self.num_parallel)]
        # as long as not yet reached number of total steps
        t = 0
        while not_done:

            if self.run_episodes:
                ready, remaining = ray.wait([b.run_n_episodes.remote(self.runner_steps) for b in runner_boxes], num_returns=self.remote_min_returns, timeout=self.remote_time_out)
            else:
                ready, remaining = ray.wait([b.run_n_steps.remote(self.runner_steps) for b in runner_boxes], num_returns=self.remote_min_returns, timeout=self.remote_time_out)

            # returns list of tuples (data_agg, index)
            returns = ray.get(ready)
            results = []
            indexes = []
            for r in returns:
                result, index = r
                results.append(result)
                indexes.append(index)

            # store data from dones
            if do_print: print(f'iteration: {t}, storing results of {len(results)} runners')
            not_done = self._store(results)
            # get boxes that are alreadey done
            accesed_mapping = map(runner_boxes.__getitem__, indexes)
            dones = list(accesed_mapping)
            # concatenate dones and not dones
            runner_boxes = dones + runner_boxes
            t += 1

        return self.data

    # stores results and asserts if we are done
    def _store(self, results):
        not_done = True
        # results is a list of dctinaries
        assert self.data.keys() == results[0].keys(), "data keys and return keys do not matach"

        for r in results:
            for k in self.data.keys():
                self.data[k].extend(r[k])

        # stop if enought data is aggregated
        if len(self.data['terminal']) > self.total_steps:
            not_done = False

        return not_done


    def sample_dictionary_of_datasets(self, sample_size, from_buffer=True):
        # sample from buffer
        if from_buffer:
            dataset_dict = self.buffer.sample_dictionary_of_datasets(sample_size)

        # else create new data
        else:
            # save old sepcification
            old_total_steps = self.total_steps
            # set to sampling size
            self.total_steps = sample_size
            self.get_data()
            dataset_dict = {}
            for k in self.data.keys():
                dataset_dict[k] = tf.data.Dataset.from_tensor_slices(self.data[k])
            # restore old specification
            self.total_steps = old_total_steps
        return dataset_dict


    def sample_dataset(self, sample_size, from_buffer=True):
        # sample from boffer
        if from_buffer:
            dataset, keys = self.buffer.sample_dataset(sample_size)
        # else create new data
        else:
            # save old specification
            old_total_steps = self.total_steps
            # set to sampling size
            self.total_steps = sample_size
            self.get_data()
            datasets = []
            for k in self.data.keys():
                datasets.append(tf.data.Dataset.from_tensor_slices(self.data[k]))
            dataset = tf.data.Dataset.zip(tuple(datasets))
            keys = self.data.keys()
            # restore old specification
            self.total_steps = old_total_steps

        return dataset, keys


    def get_agent(self):

        ray.shutdown()
        # surrpressing warnings
        ray.init(log_to_driver=False)
        runner_box = RunnerBox.remote(Agent, self.model, self.environment_name, runner_position=0, returns=self.returns, **self.kwargs)
        agent_kwargs = ray.get(runner_box.get_agent_kwargs.remote())
        agent = Agent(self.model, **agent_kwargs)

        return agent


    def set_agent(self, new_weights):
        self.kwargs['weights'] = new_weights

    def set_temperature(self, temperature):
        self.kwargs['temperature'] = temperature


    def initilize_buffer(self, size, optim_keys):
        self.buffer = Replay_buffer(size, optim_keys)

    def store_in_buffer(self, data_dict):
        self.buffer.put(data_dict)

    def test(self, test_steps, test_episodes=100, evaluation_measure='time', render=False, do_print=False):

        env = gym.make(self.environment_name)
        agent = self.get_agent()

        return_time = False
        return_reward = False

        if evaluation_measure == 'time':
            return_time = True
            time_steps = []
        elif evaluation_measure == 'reward':
            return_reward = True
            rewards = []
        elif evaluation_measure == 'time_and_reward':
            return_time = True
            return_reward = True
            time_steps = []
            rewards = []
        else: raise f"unrceognized evaluation measure: {evaluation_measure} \n Change to 'time', 'reward' or 'time_and_reward'."

        for _ in range(test_episodes):

            state_new = np.expand_dims(env.reset(), axis=0)
            if return_reward:
                reward_per_episode = []

            for t in range(test_steps):
                if render: env.render()
                state = state_new
                action = agent.act(state).numpy()
                state_new, reward, done, info = env.step(action)
                state_new = np.expand_dims(state_new, axis=0)
                if return_reward:
                    reward_per_episode.append(reward)
                if done:
                    if return_time:
                        time_steps.append(t)
                    if return_reward:
                        rewards.append(np.mean(reward_per_episode))
                    break
                if (t == test_steps):
                    if return_time:
                        time_steps.append(t)
                    if return_reward:
                        rewards.append(np.mean(reward_per_episode))
                    break

        env.close()
        # this is not pretty but it works
        if return_time:
            #avg_time = np.mean(time_steps)
            if do_print: print(f"Episodes finished after a mean of {np.mean(time_steps)} timesteps")
            if not return_reward:
                return time_steps
        if return_reward:
            #avg_reward = np.mean(rewards)
            if do_print: print(f"Episodes finished after a mean of {np.mean(rewards)} accumulated reward")
            if not return_time:
                return rewards
        return time_steps, rewards

    def initialize_aggregator(self, path, saving_after=10, aggregator_keys=['loss']):
        self.agg = Smoothing_aggregator(path, saving_after, aggregator_keys)

    def update_agg(self, **kwargs):
        self.agg.update(**kwargs)
