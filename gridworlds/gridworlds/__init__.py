from gym.envs.registration import register
from gridworlds.envs.gridworld import GridWorld

register(
    id='gridworld-v0',
    entry_point='gridworlds.envs:GridWorld',
    max_episode_steps=100000,
)
