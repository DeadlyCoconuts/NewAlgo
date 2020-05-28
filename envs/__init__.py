from envs.maze_env import *
from envs.maze_view_2d import MazeView2D


from gym.envs.registration import register


"""
Preset Mazes
"""
register(
    id='maze-preset-3x3-kpi-v0',
    entry_point='envs:MazeEnvPreset3x3',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'kpi',
            'random_reward': False}
)

register(  # Random rewards
    id='maze-preset-3x3-kpi-v1',
    entry_point='envs:MazeEnvPreset3x3',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'kpi',
            'random_reward': False}
)


register(
    id='maze-preset-3x3-exploration-v0',
    entry_point='envs:MazeEnvPreset3x3',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'exploration',
            'random_reward': False}
)

register(
    id='maze-preset-5x5-kpi-v0',
    entry_point='envs:MazeEnvPreset5x5',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'kpi',
            'random_reward': True}
)

register(  # Random rewards
    id='maze-preset-5x5-kpi-v1',
    entry_point='envs:MazeEnvPreset5x5',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'kpi',
            'random_reward': False}
)


register(
    id='maze-preset-5x5-exploration-v0',
    entry_point='envs:MazeEnvPreset5x5',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'exploration',
            'random_reward': False}
)

register(
    id='maze-preset-10x10-kpi-v0',
    entry_point='envs:MazeEnvPreset10x10',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'kpi',
            'random_reward': True}
)

register(  # Random rewards
    id='maze-preset-10x10-kpi-v1',
    entry_point='envs:MazeEnvPreset10x10',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'kpi',
            'random_reward': False}
)


register(
    id='maze-preset-10x10-exploration-v0',
    entry_point='envs:MazeEnvPreset10x10',
    max_episode_steps=1000000,
    kwargs={'reward_type': 'exploration',
            'random_reward': False}
)