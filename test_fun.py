import numpy as np
from really.agg import Smoothing_aggregator
import os

path = os.getcwd()+"/progress"

agg = Smoothing_aggregator(path=path, saving_after=10, aggregator_keys=['loss', 'reward'])

for i in range(1000):
    agg.update(loss=[np.random.rand(100) for _ in range(2)], reward=np.random.rand(100))
