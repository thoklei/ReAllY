import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
from really import SampleManager


class MyModel(tf.keras.Model):

    def __init__(self, output_units=2):

        super(MyModel, self).__init__()
        self.layer = tf.keras.layers.Dense(output_units)
        self.layer2 = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        x = self.layer(x_in)
        v = self.layer2(x)
        output['output'] = x
        output['value_estimate'] = v
        return output


class ModelContunous(tf.keras.Model):

    def __init__(self, output_units=2):

        super(ModelContunous,self).__init__()

        self.layer_mu = tf.keras.layers.Dense(output_units)
        self.layer_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        mus = self.layer_mu(x_in)
        sigmas = tf.exp(self.layer_sigma(x_in))
        v = self.layer_v(x_in)
        output['mu'] = mus
        output['sigma'] = sigmas
        output['value_estimate'] = v

        return output


if __name__== "__main__":

    kwargs = {
        'model' : MyModel,
        'environment_name' :'CartPole-v0',
        'num_parallel' :5,
        'total_steps' :100,
        'action_sampling_type' :'thompson',
        'num_episodes': 20,

    }

    manager = SampleManager(**kwargs)
    saving_path = os.getcwd()+'/progress_test'

    buffer_size = 500
    test_steps = 50
    epochs = 10
    sample_size = 10
    optim_batch_size = 8
    gamma = 0.95
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.MSE
    saving_after = 2

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ['state', 'action', 'reward', 'state_new', 'terminal']

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)
    agent = manager.get_agent()

    # initilize progress aggregator
    manager.initialize_aggregator(path=saving_path, saving_after=saving_after, aggregator_keys=['loss', 'time_steps'])

    # initial testing:
    print('test before training: ')
    manager.test(test_steps, do_print=True)

    for e in range(epochs):

        # training core

        # experience replay
        print('collecting experience..')
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        experience_dict = manager.sample_dictionary_of_datasets(sample_size)
        # batch datasets
        for k in experience_dict:
            experience_dict[k] = experience_dict[k].batch(optim_batch_size)


        print('optimizing...')

        # TODO: iterate through your datasets

        # TODO: optimize agent

        dummy_losses = [np.random.rand(10) for _ in range(10)]

        new_weights = agent.model.get_weights()
        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_agg(loss=dummy_losses, time_steps=time_steps)
        print(f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"   )

    print('done')
    print('testing optimized agent')
    manager.test(test_steps, render=True)
