import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds.envs.gridworld import GridWorld

"""
Solving Gridworld with Q-learning.
"""

class TabularQ(object):


    def __init__(self, h, w, action_space):
        """
        Constructor for Tabular Q Model.

        h = height of the gridworld
        w = width of the gridworld
        action_space = the number of actions that can be performed 
        """

        self.action_space = action_space
        self.table = np.zeros((action_space, h, w))
        

    def __call__(self, state):
        """
        Given a state, outputs an array of Q-values for the actions.

        state = the state to be interpreted
        """
        state = np.squeeze(state)
        x,y = state
        x = int(x)
        y = int(y)

        output = {}
        output["q_values"] = np.expand_dims(self.table[:, x,y], axis=0)

        return output
        

    def get_weights(self):
        """
        Returns the Q-table.
        """
        return self.table.copy()


    def set_weights(self, q_vals):
        """
        Updates the values stored in the Q-table.

        q_vals = the new q values
        """
        self.table = q_vals


if __name__ == "__main__":

    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 5,
        "width": 5,
        "block_position": (3,3),
        "action_dict": action_dict,
        "start_position": (0, 0),
        "reward_position": (4, 4),
    }

    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 20,
        "model_kwargs": model_kwargs,
        "env_kwargs": env_kwargs
    }

    # initializing ray
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    saving_path = os.getcwd() + "/progress_test"

    epochs = 10 
    buffer_size = 5000
    test_steps = 50
    sample_size = 1000
    saving_after = 5

    alpha = 0.1
    gamma = 0.95

    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initialize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["time_steps"]
    )

    print("Testing before training... ")
    manager.test(
        max_steps=10,
        test_episodes=3,
        render=False,
        do_print=True
    )

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # experience replay
        print("Collecting experience...")
        data = manager.get_data(total_steps=1000)
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)

        print("Optimizing...")

        # iterating through dataset
        old_table = agent.get_weights()
        for s, a, r, n in zip(sample_dict['state'], sample_dict['action'], sample_dict['reward'], sample_dict['state_new']):
            
            s_x, s_y = s # unpacking state
            n_x, n_y = n # unpacking new state
            
            # Apply Q-learning formula
            old_table[a, s_x, s_y] += alpha * (r + gamma * np.max(old_table[:, n_x, n_y]) - old_table[a, s_x, s_y])

        # set new weights
        manager.set_agent(old_table)

        # get new weights
        agent = manager.get_agent()
        
        time_steps = manager.test(test_steps)

        # update aggregator
        manager.update_aggregator(time_steps=time_steps)
        
        print(f"epoch ::: {e}  avg env steps ::: {np.mean(time_steps)}")

    print("Done!")

    print("Testing optimized agent...")
    manager.test(test_steps, test_episodes=3, render=True)