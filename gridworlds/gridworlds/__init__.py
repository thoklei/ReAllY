from gym.envs.registration import register
from gridworlds.envs.gridworld import GridWorld
from gridworlds.envs.gridworld_global import GridWorld_Global

register(
    id="gridworld-v0",
    entry_point="gridworlds.envs:GridWorld",
    max_episode_steps=100000,
)
register(
    id="gridworld-v1",
    entry_point="gridworlds.envs:GridWorld_Global",
    max_episode_steps=100000,
)