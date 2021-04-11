import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path


cases = ["PPO", "RND"]
envs = ["Bipedal_easy", "Lunar"]
authors = ["Thomas", "Nicolai", "Malte"]

for env_name in envs:
    dfs_env = {}
    for case in cases:
        dfs_env[case] = []
        for author in authors:
            file_name = f"results/{case}_{env_name}_{author}_1.csv"
            if not os.path.isfile(file_name):
                print(f"FAILED: {file_name} does not exist")
            else:
                print(f"SUCCESS: found file {file_name}")
                dfs_env[case].append(pd.read_csv(file_name, index_col='epoch'))


    fig, ax = plt.subplots()
    window_size = 30
    for df in dfs_env["PPO"]:
        df = df.truncate(after=400) # truncate df
        df['avg_reward'] = df['reward'].rolling(window=window_size).mean() # calc running average
        max_index = int(df[['avg_reward']].idxmax())
        max_value = df[['avg_reward']].max()
        df1 = df['avg_reward'].iloc[:max_index]
        df2 = df['avg_reward'].iloc[max_index:]
        ax.plot(df1, label="PPO", alpha=0.6, linestyle='-', color="g")
        ax.plot(df2, alpha=0.6, linestyle='--', color="g")
        ax.plot(max_index, max_value, alpha=0.6, marker='o', color="g")
    for df in dfs_env["RND"]:
        df = df.truncate(after=400) # truncate df
        df['avg_reward'] = df['reward'].rolling(window=window_size).mean() # calc running average
        max_index = int(df[['avg_reward']].idxmax())
        max_value = df[['avg_reward']].max()
        df1 = df['avg_reward'].iloc[:max_index]
        df2 = df['avg_reward'].iloc[max_index:]
        ax.plot(df1, label="PPO", alpha=0.6, linestyle='-', color="r")
        ax.plot(df2, alpha=0.6, linestyle='--', color="r")
        ax.plot(max_index, max_value, alpha=0.6, marker='o', color="r")
    ax.set_title(env_name)
    ax.legend()
    plt.show()
