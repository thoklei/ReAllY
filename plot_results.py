import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

bipedal = "BipedalWalker-v3"
lunar = "LunarLanderContinuous-v2"

cases = ["PPO", "RND"]
envs = ["Bipedal_easy", "Lunar"]
authors = ["Thomas", "Nicolai", "Malte"]

for env_name in envs:
    for case in cases:
        file_names = []
        for author in authors:
            file_name = f"results/{case}_{env_name}_{author}_1.csv"
            # check if csv file exists
            if not os.path.isfile(file_name):
                print(f"FAILED: file {file_name} does not exist")
            else:
                print(f"SUCCESS: file {file_name} found")
                file_names.append(file_name)

        if file_names:
            first = True
            for file in file_names:
                df = pd.read_csv(file, index_col='epoch')
                if 'rnd_loss' in df.columns:
                    df = df.drop(columns=['rnd_loss'])
                if first:
                    ax = df.plot(subplots=True)
                    first = False
                else:
                    df.plot(ax=ax, subplots=True, legend=False)
            plt.show()

                
