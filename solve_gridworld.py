import gym
import numpy as np
import tensorflow as tf
import ray
from really import SampleManager

#import sys
#sys.path.append('gridworlds/gridworlds/envs')
from gridworld import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):

    def __init__(self, h, w, action_space):
        self.action_space = action_space
        self.table = np.zeros((h, w, action_space))
        

    def __call__(self, state):

        if isinstance(state, tf.python.framework.ops.EagerTensor):
            x,y = state[0]
            x = int(x.numpy())
            y = int(y.numpy())
        else:
            x,y = state[0]

        self.state_x = x 
        self.state_y = y
        
        print("x, y in call: ",x,y)

        output = {}
        output["q_values"] = tf.constant([self.table[x,y,:]])

        # TODO return v_estimate
        return output
        

    def get_weights(self):
        return self.table

    def set_weights(self, q_vals):
        print("Q-vals in set_weights: ", q_vals)
        self.table = q_vals


    # what else do you need?


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 1,
        "total_steps": 10,
        "model_kwargs": model_kwargs
        # and more
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    print("Agent: ",manager.get_agent())

    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )

    # do the rest!!!!
    epochs = 3

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        # TODO: iterate through your datasets

        # TODO: optimize agent

        dummy_losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]

        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

