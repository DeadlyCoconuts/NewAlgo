import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .maze_view_2d import MazeView2D
import math
from util import *
import pygame

class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, reward_type=None, maze_file=None, maze_size=None, mode=None, enable_render=True, random_reward=False):
        self.viewer = None
        self.enable_render = enable_render

        if maze_file:
            #self.__num_rewards = 2
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640),
                                        #num_rewards=self.__num_rewards,
                                        enable_render=enable_render)
            self.__num_rewards = self.maze_view.maze.num_rewards
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size) / 3))
                self.__num_rewards = 3
            else:
                has_loops = False
                num_portals = 0
                self.__num_rewards = int(round(min(maze_size) / 3 + 2))

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        num_rewards=self.__num_rewards,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # retrieve the state space
        self.__list_state = list(range(0, self.maze_size[0] * self.maze_size[1]))

        self.__state_to_list_idx = dict()
        temp_idx = 0

        for y in range(self.maze_size[1]):
            for x in range(self.maze_size[0]):
                self.__state_to_list_idx[(x, y)] = temp_idx
                temp_idx += 1

        # retrieve the action space
        self.__list_action = dict()
        for state in self.__list_state:
            self.__list_action[state] = np.arange(4)

        # retrieve the transition probability kernel
        self.__transition_p_kernel = dict()
        for y in range(self.maze_size[1]):
            for x in range(self.maze_size[0]):
                for action in self.maze_view.maze.DIRECTIONAL_KEYS.keys():
                    position = (x, y)
                    transition_probability_vector = np.zeros(self.maze_size[0] * self.maze_size[1])
                    if self.maze_view.maze.is_open(position, action):
                        position += np.array(self.maze_view.maze.COMPASS[action])
                    transition_probability_vector[self.maze_size[1] * position[1] + position[0]] = 1
                    self.__transition_p_kernel[(self.state_to_list_idx[(x, y)], self.maze_view.maze.DIRECTIONAL_KEYS[action])] = \
                        transition_probability_vector

        # set up reward dictionary
        self.__reward_type = reward_type
        self.reward_dict = dict()
        self.num_dim = math.ceil(self.__num_rewards)  # * 0.8)

        # set random reward vectors for each reward as well as targets
        self.v_mean = dict()  # contains state-action pairs
        if reward_type == "exploration":
            self.target = np.zeros(self.maze_size[0] * self.maze_size[1])
            target_value = 1/len(self.maze_view.maze.rewards)
            for y in range(0, self.maze_size[0]):
                for x in range(0, self.maze_size[1]):
                    # set rewards
                    reward_vector = np.zeros(self.maze_size[0] * self.maze_size[1])
                    reward_vector[y * self.maze_size[0] + x] = 1
                    self.reward_dict[(x, y)] = reward_vector
                    # set v_mean
                    for action in range(4):
                        self.v_mean[self.__state_to_list_idx[(x, y)], action] = reward_vector
                    # set target
                    if (x, y) in self.maze_view.maze.rewards:
                        self.target[y * self.maze_size[0] + x] = target_value
        elif reward_type == "kpi":
            # set rewards
            target_matrix = np.zeros((1, self.num_dim))
            idx = 0
            for y in range(0, self.maze_size[0]):
                for x in range(0, self.maze_size[1]):
                    # set rewards
                    reward_vector = np.zeros(self.num_dim)
                    if (x, y) in self.maze_view.maze.rewards:
                        if random_reward is True:
                            row_ind = np.random.choice(self.num_dim, math.ceil(self.num_dim / 2), replace=False)
                            for i in row_ind:
                                reward_vector[i] = np.random.uniform(0.3, 1)
                        else:
                            reward_vector[idx] = 1
                            idx += 1
                        if idx == 3:
                            reward_vector = reward_vector / np.sum(reward_vector)  # normalise the vector
                    self.reward_dict[(x, y)] = reward_vector

                    # prepare target_vector
                    if (x, y) != (0, 0):
                        target_matrix = np.concatenate((target_matrix, reward_vector.reshape((1, self.num_dim))))

            # set v _mean
            for y in range(0, self.maze_size[0]):
                for x in range(0, self.maze_size[1]):
                    state_k = (x, y)
                    state = self.state_to_list_idx[(x, y)]
                    if state_k in self.maze_view.maze.rewards:
                        for action in range(4):
                            self.v_mean[(state_k, action)] = np.dot(self.transition_p_kernel[(state, action)], target_matrix)
                    else:
                        for action in range(4):
                            self.v_mean[(state_k, action)] = np.zeros(self.num_dim)

            """
            for state in range(self.maze_size[0] * self.maze_size[1]):
                    for action in range(4):
                        self.v_mean[(state, action)] = np.dot(self.transition_p_kernel[(state, action)], target_matrix)
            """

            # set target
            self.target = np.dot(np.ones(self.maze_size[0] * self.maze_size[1]), target_matrix) / self.num_dim

            """
            for reward_location in self.maze_view.maze.rewards:
                row_ind = np.random.choice(self.num_dim, math.ceil(self.num_dim / 2), replace=False)
                reward_vector = np.zeros(self.num_dim)
                for i in row_ind:
                    reward_vector[i] = np.random.uniform(0.3, 1)
                reward_vector = reward_vector / np.sum(reward_vector)  # normalise the vector
                self.reward_dict[reward_location] = reward_vector
            """
        # retrieve reward dimension
        self.__reward_dim = self.target.shape[0]

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2 * len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def algo_step(self, action):  # used only by core.py to bypass Env function constraints for scalar reward
        if isinstance(action, int):
            real_reward = self.reward_dict[tuple(self.maze_view.robot)]
        else:
            real_reward = self.reward_dict[tuple(self.maze_view.robot)]
        return real_reward

    def step(self, action):
        if isinstance(action, int):
            #real_reward = self.reward_dict[tuple(self.maze_view.robot)]
            self.maze_view.move_robot(self.ACTION[action])
        else:
            #real_reward = self.reward_dict[tuple(self.maze_view.robot)]
            self.maze_view.move_robot(action)

        """
        if np.array_equal(self.maze_view.robot, self.maze_view.goal):  # TODO: Add cases for rewards
            reward = 1
            done = True
        if self.maze_view.maze.is_reward(self.maze_view.robot):
            reward = self.reward_dict[tuple(self.maze_view.robot)]
            done = False
        else:
            # reward = -0.1 / (self.maze_size[0] * self.maze_size[1])
            reward = np.zeros(self.num_dim)
            done = False
        """

        done = False

        self.state = self.maze_view.robot

        info = {}

        reward = 0 # dummy reward to satisfy StatsRecorder requirements

        return self.state, reward, done, info

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)

    def save_maze(self):
        pygame.image.save(self.maze_view.screen, 'testcrap.jpeg')

    @property
    def num_rewards(self):
        return self.__num_rewards

    @property
    def reward_type(self):
        return self.__reward_type

    @property
    def list_state(self):
        return self.__list_state

    @property
    def state_to_list_idx(self):
        return self.__state_to_list_idx

    @property
    def list_action(self):
        return self.__list_action

    @property
    def transition_p_kernel(self):
        return self.__transition_p_kernel

    @property
    def reward_dim(self):
        return self.__reward_dim

class MazeEnvPreset3x3(MazeEnv):
    def __init__(self, enable_render=True, **kwargs):
        super(MazeEnvPreset3x3, self).__init__(maze_file="maze2d_001.npy", enable_render=enable_render,
                                               reward_type=kwargs.get("reward_type"),
                                                random_reward=kwargs.get("random_reward"))

class MazeEnvPreset5x5(MazeEnv):
    def __init__(self, enable_render=True, **kwargs):
        super(MazeEnvPreset5x5, self).__init__(maze_file="maze2d_002.npy", enable_render=enable_render,
                                               reward_type=kwargs.get("reward_type"),
                                                random_reward=kwargs.get("random_reward"))

class MazeEnvPreset10x10(MazeEnv):
    def __init__(self, enable_render=True, **kwargs):
        super(MazeEnvPreset10x10, self).__init__(maze_file="maze2d_003.npy", enable_render=enable_render,
                                               reward_type=kwargs.get("reward_type"),
                                                random_reward=kwargs.get("random_reward"))