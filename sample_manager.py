import os
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ray
import tensorflow as tf
from agent import Agent
from runner_box import RunnerBox


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
        value_estimate: boolean, if agent returns value estimate, defaults to false
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
                kwargs['action_sampling_type'] = 'thompson'


        # chck return specifications
        for r in returns:
            if r not in ['log_prob', 'monte_carlo', 'value_estimate']:
                print(f'unsuppoerted return key: {r}')
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
            kwargs.pop('num_episodes')
        elif 'num_steps' in kwargs.keys():
            sef.runner_steps = kwargs['num_steps']
            self.run_episodes = False
            kwargs.pop('num_steps')

        # check for remote process specifications
        if 'remote_min_returns' in kwargs.keys():
            self.remote_min_returns = kwargs['remote_min_returns']
            kwargs.pop('remote_min_returns')
        else:
            # defaults to 10% of remote runners, but minimum 1
            self.remote_min_returns = max([int(0.1 * self.num_parallel),1])

        if 'remote_time_out' in kwargs.keys():
            self.remote_time_out = kwargs['remote_time_out']
            kwargs.pop('remote_time_out')
        else:
            # defaults to None, i.e. wait for remote_min_returns to be returned irrespective of time
            self.remote_time_out = None


    def _get_data(self):
        not_done = True
        ray.init(log_to_driver=False)

        # create list of runnor boxes
        runner_boxes = [RunnerBox.remote(Agent, self.model, self.environment_name, runner_position=i, returns=self.returns, **self.kwargs) for i in range(self.num_parallel)]
        # as long as not yet reached number of total steps
        t = 0
        while not_done:

            if self.run_episodes:
                print('we are here')
                ready, remaining = ray.wait([b.run_n_episodes.remote(self.runner_steps) for b in runner_boxes], num_returns=self.remote_min_returns, timeout=self.remote_time_out)
            else:
                ready, remaining = ray.wait([b.run_n_steps.remote(self.runner_steps) for b in runner_boxes], num_returns=self.remote_min_returnso, timeout=self.remote_time_out)

            # returns list of tuples (data_agg, index)
            returns = ray.get(ready)
            results = []
            indexes = []
            for r in returns:
                result, index = r
                results.append(result)
                indexes.append(index)

            # store data from dones
            print(f'iteration: {t}, storing results of {len(results)} runners')
            not_done = self._store(results)
            # get boxes that are alreadey done
            accesed_mapping = map(runner_boxes.__getitem__, indexes)
            dones = list(accesed_mapping)
            # concatenate dones and not dones
            runner_boxes = dones + runner_boxes
            t += 1
        # return data

    # sores results and asserts if we are done
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

    def create_dictionary_of_datasets(self, new=True):
        if new: self._get_data()
        dataset_dict = {}
        for k in self.data.keys():
            dataset_dict[k] = tf.data.Dataset.from_tensor_slices(self.data[k])
        return dataset_dict

    def create_dataset(self, new=True):
        if new: self._get_data()
        datasets = []
        for k in self.data.keys():
            datasets.append(tf.data.Dataset.from_tensor_slices(self.data[k]))
        dataset = tf.data.Dataset.zip(tuple(datasets))


        return dataset, self.data.keys()
