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
    window_size = 25

    # colors_red = ['r', 'r', 'r']
    colors_red = ['#CD0000', '#CD0000', '#CD0000']
    # colors_green = ['#00CD00', '#008B45', '#006400']
    colors_green = ['g', 'g', 'g']

    put_label = True
    for i, df in enumerate(dfs_env["PPO"]):
        df = df.truncate(after=400) # truncate df after 400
        df['avg_reward'] = df['reward'].rolling(window=window_size).mean() # calc running average
        max_index = int(df[['avg_reward']].idxmax())
        max_value = df[['avg_reward']].max()
        df1 = df['avg_reward'].iloc[:max_index]
        df2 = df['avg_reward'].iloc[max_index:]
        if put_label:
            ax.plot(df1, label="PPO", alpha=0.7, linestyle='-', color=colors_green[i])
            put_label = False
        else:
            ax.plot(df1, alpha=0.7, linestyle='-', color=colors_green[i])
        ax.plot(df2, alpha=0.4, linestyle='--', color=colors_green[i])
        ax.plot(max_index, max_value, alpha=0.8, marker='o', color=colors_green[i])
    put_label = True
    for i, df in enumerate(dfs_env["RND"]):
        df = df.truncate(after=400) # truncate df
        df['avg_reward'] = df['reward'].rolling(window=window_size).mean() # calc running average
        max_index = int(df[['avg_reward']].idxmax())
        max_value = df[['avg_reward']].max()
        df1 = df['avg_reward'].iloc[:max_index]
        df2 = df['avg_reward'].iloc[max_index:]
        if put_label:
            ax.plot(df1, label="RND", alpha=0.7, linestyle='-', color=colors_red[i])
            put_label = False
        else:
            ax.plot(df1, alpha=0.7, linestyle='-', color=colors_red[i])
        ax.plot(df2, alpha=0.4, linestyle='--', color=colors_red[i])
        ax.plot(max_index, max_value, alpha=0.8, marker='o', color=colors_red[i])

    if env_name == "Lunar":
        ax.set_title("Performance: LunarLander")
    if env_name == "Bipedal_easy":
        ax.set_title("Performance: Bipedal Walker")
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"reward\n(rolling average with a window of {window_size} epochs)")
    plt.tight_layout()
    plt.savefig(f'results/{env_name}.png')
    plt.show()

