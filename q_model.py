
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import gym
import ray

from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets

class QTable(tf.keras.Model):

    def __init__(self):
        super(QTable, self).__init__()


    def call(self, x_in):

        output = {}
        output['q_values'] = np.array([0.5])#np.array([0.1,0.4,0.4,0.1])
        #output['v_estimate'] = 0.5

        return output