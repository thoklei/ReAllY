import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path


cases = ["PPO", "RND"]
envs = ["Bipedal_easy", "Lunar"]
authors = ["Thomas", "Nico", "Malte"]

for env_name in envs:
    dfs_env = {}
    for case in cases:
        dfs_env[case] = []
        for author in authors:
            file_name = f"results/{case}_{env_name}_{author}_1.csv"
            if not os.path.isfile(file_name):
                print(f"file {file_name} does not exist")
            else:
                print(f"found file {file_name}")
                dfs_env[case].append(pd.read_csv(file_name, index_col='epoch'))


    fig, ax = plt.subplots()
    for df in dfs_env["PPO"]:
        ax.plot(df["reward"], label="PPO", alpha=0.6)
    for df in dfs_env["RND"]:
        ax.plot(df["reward"], label="RND", alpha=0.6)
    ax.set_title(env_name)
    ax.legend()
    plt.show()

