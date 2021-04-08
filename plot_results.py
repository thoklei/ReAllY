import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

bipedal = "BipedalWalker-v3"
lunar = "LunarLanderContinuous-v2"
envs = [bipedal, lunar]

for env_name in envs:
    results_rnd = env_name + '/results_rnd'
    results_base = env_name + '/results_base'
    for run in [results_base, results_rnd]:
        file_names = []
        for i in [1, 2, 3]:
            file_name = run + str(i) + ".csv"
            # check if csv file exists
            if not os.path.isfile(file_name):
                print(f"file {file_name} does not exist")
            else:
                file_names.append(file_name)

        if file_names:
            # Read first CSV file
            df1 = pd.read_csv(file_names[0], index_col=0)
            # plot dataframe and create figure
            ax = df1.plot(subplots=True)
            # add more dataframes to figure
            for j in range(len(file_names)-1):
                pd.read_csv(file_names[j+1], index_col=0).plot(ax=ax, subplots=True, legend=False)
            plt.show()

                
