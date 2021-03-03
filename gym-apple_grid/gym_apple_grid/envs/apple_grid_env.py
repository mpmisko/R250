import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
from agent import Agent

ENV_DATA = {
    'empty' : 0,
    'apple' : 1,
    '0' : 2,
    '1' : 3
}

RENDER_DATA = {
    0 : np.array([0, 0, 0]), # Black
    1 : np.array([0, 255, 0]), # Green
    2 : np.array([255, 0, 0]), # Red
    3 : np.array([0, 0, 255]), # Blue
}

class GridEnvConfig:
    def __init__(self,
                dimensions,
                num_apples,
                num_actors,
                obs_window_size):
        self.dimensions = dimensions
        self.num_apples = num_apples
        self.num_actors = num_actors
        self.obs_window_size = obs_window_size
    
class AppleGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, is_rendering=False):
        self.config = config
        self.is_rendering = is_rendering
        self.grid, self.agents =  self._init_map()
        self.curr_step = 0
        self.apples_eaten = 0
        

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def _sample_random_location(self):
        x = np.random.randint(low=0, high=self.config.dimensions[0], size=1)
        y = np.random.randint(low=0, high=self.config.dimensions[1], size=1)
        return x, y

    def _init_map(self):
        grid = np.array(self.config.dimensions)
        
        # Place apples
        apple_count = 0
        while apple_count < self.config.num_apples:
            x, y = self._sample_random_location()
            if not grid[x][y] == ENV_DATA['empty']:
                continue
                
            grid[x][y] = ENV_DATA['apple']
            apple_count += 1

        # Place and initialise actors
        actor_states = {}
        actor_count = 0
        while actor_count < self.config.num_actors:
            x, y = self._sample_random_location()
            if not grid[x][y] == ENV_DATA['empty']:
                continue
            
            actor_states[actor_count] = Agent(actor_count, np.array([x, y]), 'up')
            grid[x][y] = ENV_DATA[str(actor_count)]
            actor_count += 1

        return grid, actor_states

    def step(self, actions):
        assert len(actions) == self.config.num_actors

        indexed_actions = [(i, actions[i]) for i in range(len(actions))]
        random.shuffle(indexed_actions)

        observations = [None] * len(actions)
        rewards = [None] * len(actions)
        info = [None] * len(actions)

        # Apply actions and compute rewards
        for i, action in indexed_actions:
            agent = self.agents[i]
            action = agent.logit_to_action(action)
            self.grid, reward = agent.apply_action(action, self.grid, self.agents, curr_step)
            rewards[i] = reward
            self.apples_eaten += 1 if reward > 0 else 0

        for idx, agent in self.agents.items():
            observations[idx] = agent.grid_to_observation(self.grid, RENDER_DATA)
        
        self.curr_step += 1
        dones = [self.curr_step >= 1000] * len(actions)

        return observations, rewards, dones, info


    def reset(self):
        self.grid, self.agents =  self._init_map()
        self.curr_step = 0
        self.apples_eaten = 0

    def _map_to_colors(self):
        img = np.zeros([*self.grid.shape, 3])
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                img[i][j] = RENDER_DATA[self.grid[i][j]]

        return img

    def render(self, mode='human'):
        rgb_arr = self._map_to_colors(self.grid)
        plt.imshow(rgb_arr, interpolation='nearest')
