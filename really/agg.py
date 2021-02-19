import numpy as np
import matplotlib.pyplot as plt
import os, logging
# only print error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Smoothing_aggregator:

    def __init__(self, path, saving_after=100, aggregator_keys = ['loss'], max_size=2):
        self.aggregator_size = 0
        self.aggregator_max_size = max_size
        self.aggregator = {}
        self.aggregator_vals = {}

        for k in aggregator_keys:
            self.aggregator[k] = []
            self.aggregator_vals[k] = []

        self.path = path
        self.saving_after = saving_after
        self.epoch = 0
        self.reached_size=False

    def update(self, **kwargs):

        increased = False
        saved = False
        for k in kwargs.keys():
            if k in self.aggregator.keys():
                if k=='loss':
                    kwargs[k] = [np.squeeze(a) for a in kwargs[k]]
                    self.aggregator[k] = np.concatenate([self.aggregator[k], np.concatenate(kwargs[k])]).tolist()
                else: self.aggregator[k].append(kwargs[k])
                increased = True
            else:
                print(f"unsupported aggregator key: {k}, aggregator was only initialized with the keys {self.aggregator.keys()}")
                raise KeyError

        if increased: self.aggregator_size+=1

        if self.aggregator_size >= self.aggregator_max_size:
            for k in kwargs.keys():
                self.aggregator_vals[k].append(np.mean(np.asarray(self.aggregator[k])))
                self.aggregator[k] = []
            self.aggregator_size = 0
            if not(self.reached_size): self.reached_size=True
        if (self.epoch%self.saving_after==0) and self.reached_size:
            self.save_graphic()
        self.epoch += 1


    def save_graphic(self):

        keys = list(self.aggregator_vals.keys())
        number_of_subplots = len(keys)
        plt.clf()
        plt.suptitle(f'training process after {self.epoch} training epochs')
        for i,v in enumerate(range(number_of_subplots)):
            v+=1
            ax1 = plt.subplot(number_of_subplots,1,v, label=keys[i])
            ax1.plot(self.aggregator_vals[keys[i]])
            ax1.set_ylabel(keys[i])

        plt.xlabel(f"1 step aggregated over {self.aggregator_max_size} * aggregated data per epoch")
        plt.savefig(f"{self.path}/progress_{self.epoch}.png")
