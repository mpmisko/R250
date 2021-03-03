import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt

ACTIONS = {
    'left' : [-1, 0],
    'right' : [1, 0],
    'up' : [0, 1],
    'down' : [0, -1],
    'noop' : [0, 0],
    'turn_counterclockwise' : [[0, -1], 
                                [1, 0]],
    'turn_clockwise' : [[0, 1], 
                        [-1, 0]],
    'shoot' : []
}

ORIENTATIONS = {'left': [-1, 0],
                'right': [1, 0],
                'up': [0, -1],
                'down': [0, 1]}

ENV_DATA = {
    'empty' : 0,
    'apple' : 1,
    '0' : 2,
    '1' : 3
}

AGENT_TO_IDX = {
    2 : 0,
    3 : 1
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
        
class Agent:
    def __init__(self, index, location, orientation):
        self.orientation = orientation
        self.location = location
        self.defreeze_delta = 5
        self.shot_time = -1
        self.index = index

    def logit_to_action(self, logits):
        idx = np.argmax(logits)
        for i, (k, v) in enumerate(ACTIONS):
            if i == idx:
                return k

        raise RuntimeError
    
    def _is_in_map(self, location, grid):
        return 0 <= location[0] < grid.shape[0] and 0 <= location[1] < grid.shape[1]

    def apply_action(self, action, grid, agents, curr_step):
        # Returns updated grid, obtained rewards

        if self.shot_time != -1 and shot_time != curr_step:
            if curr_step - self.shot_time > self.defreeze_delta:
                self.shot_time = -1
            
            return grid, 0

        if action == "shoot":
            loc = self.location + ORIENTATIONS[self.orientation]
            while self._is_in_map(loc, grid):
                content = grid[loc[0]][loc[1]]
                if content != ENV_DATA['empty'] and content != ENV_DATA['apple']:
                    agents[AGENT_TO_IDX[content]].shot_time = curr_step
                else:
                    loc += ORIENTATIONS[self.orientation]
        
        elif action.startswith('turn'):
            R = np.array(ACTIONS[action])
            new_orientation = np.matmul(R, ORIENTATIONS[self.orientation])
            for or_name, or_vals in ORIENTATIONS.items():
                if np.allclose(new_orientation, or_vals):
                    self.orientation = or_name
                    break

        else:
            next_loc = self.location + ACTIONS[action]
            
            # Out of bounds
            if not self._is_in_map(next_loc, grid):
                return grid, 0

            # Stepping on other player or noop
            content = grid[next_loc[0]][next_loc[1]]
            if not (content == ENV_DATA['empty'] or content == ENV_DATA['apple']):
                return grid, 0

            # Can make step
            grid[self.location[0]][self.location[1]] = ENV_DATA['empty']
            reward = 1 if content == ENV_DATA['apple'] else 0
            grid[self.location[0]][self.location[1]] = ENV_DATA[str(self.index)]
            self.location = next_loc
            
            return grid, reward

class AppleGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, is_rendering=False):
        self.config = config
        self.is_rendering = is_rendering
        self.grid, self.agents =  self._init_map()
        self.curr_step = 0
        self.apples_eaten = 0

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

    def _grid_to_observation(self, grid, actor):
        # Assumes window is a square withe even length sides

        len_x = self.grid.config.obs_window_size[0] + 1
        len_y = self.grid.config.obs_window_size[1] + 1

        obs = np.zeros([len_x, len_y, 3])
        center = actor.location

        for i in range(center[0] - len_x // 2, center[0] + len_x // 2 + 1):
            for j in range(center[1] - len_y // 2, center[1] + len_y // 2 + 1):
                obs_i = i - center[0] - len_x // 2
                obs_j = j - center[1] - len_y // 2

                if not 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                    obs[obs_i][obs_j] = np.array([255., 255., 255.]) # Out of bounds
                else:
                    color = RENDER_DATA[grid[i][j]]
                    obs[obs_i][obs_j] = color

        if actor.orientation == 'right':
            obs = np.rot90(obs, k=1)
        elif actor.orientation == 'down':
            obs = np.rot90(obs, k=2)
        elif actor.orientation == 'left':
            obs = np.rot90(obs, k=3)

        return obs
        

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
            observations[idx] = self._grid_to_observation(grid, agent)
        
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