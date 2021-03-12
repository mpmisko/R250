import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete
import matplotlib.pyplot as plt

ACTIONS = { 'left' : [-1, 0],
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

ACTION_ARR = ['up', 'noop', 'turn_counterclockwise', 'turn_clockwise', 'shoot']

ORIENTATIONS = {'left': np.array([-1, 0]),
                'right': np.array([1, 0]),
                'up': np.array([0, -1]),
                'down': np.array([0, 1])
}

ENV_DATA = {'empty' : 0,
            'apple' : 1,
            '0' : 2,
            '1' : 3
}

AGENT_TO_IDX = {
    2 : 0,
    3 : 1
}

class Agent:
    def __init__(self, index, location, orientation, window_size):
        self.orientation = orientation
        self.location = location
        self.defreeze_delta = 5
        self.shot_time = -1
        self.index = index
        self.obs_window_size = window_size
    
    @property
    def action_space(self):
        return Discrete(5)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(3, self.obs_window_size + 1,
                                             self.obs_window_size + 1), dtype=np.float32)

    def logit_to_action(self, logits):
        idx = np.argmax(logits)
        for i, (k, v) in enumerate(ACTIONS):
            if i == idx:
                return k

        raise RuntimeError

    def get_action(self, action):
        return ACTION_ARR[action]

    def _move_by_delta(self, location, action):
        return np.array([location[0] + action[1], location[1] + action[0]])

    def _is_in_map(self, location, grid):
        return 0 <= location[0] < grid.shape[0] and 0 <= location[1] < grid.shape[1]

    def apply_action(self, action, grid, agents, curr_step):
        # Returns updated grid, obtained rewards

        if self.shot_time != -1 and self.shot_time != curr_step:
            if curr_step - self.shot_time > self.defreeze_delta:
                self.shot_time = -1
            
            return grid, 0

        if action == "shoot":
            loc = self._move_by_delta(self.location, ORIENTATIONS[self.orientation])
            while self._is_in_map(loc, grid):
                content = grid[loc[0]][loc[1]]
                if content != ENV_DATA['empty'] and content != ENV_DATA['apple']:
                    agents[AGENT_TO_IDX[content]].shot_time = curr_step
                    break
                else:
                    loc = self._move_by_delta(loc, ORIENTATIONS[self.orientation])

            return grid, 0
        
        if action.startswith('turn'):
            R = np.array(ACTIONS[action])
            new_orientation = np.matmul(R, ORIENTATIONS[self.orientation])
            for or_name, or_vals in ORIENTATIONS.items():
                if new_orientation[0] == or_vals[0] and new_orientation[1] == or_vals[1]:
                    self.orientation = or_name
                    break

            return grid, 0.0


        next_loc = self._move_by_delta(self.location,  ORIENTATIONS[self.orientation])
        
        # Out of bounds
        if not self._is_in_map(next_loc, grid):
            return grid, 0

        # Stepping on other player or noop
        content = grid[next_loc[0]][next_loc[1]]
        if not (content == ENV_DATA['empty'] or content == ENV_DATA['apple']):
            return grid, 0

        # Can make step
        grid[self.location[0]][self.location[1]] = ENV_DATA['empty']
        reward = 10 if content == ENV_DATA['apple'] else 0
        grid[next_loc[0]][next_loc[1]] = ENV_DATA[str(self.index)]
        self.location = next_loc
        
        return grid, reward

    def grid_to_observation(self, grid, render_data):
        # Assumes window is a square withe even length sides

        len_x = self.obs_window_size + 1
        len_y = self.obs_window_size + 1

        obs = np.zeros([len_x, len_y, 3])
        center = self.location

        for i in range(center[0] - len_x // 2, center[0] + len_x // 2 + 1):
            for j in range(center[1] - len_y // 2, center[1] + len_y // 2 + 1):
                obs_i = i - (center[0] - len_x // 2)
                obs_j = j - (center[1] - len_y // 2)

                if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
                    obs[obs_i][obs_j] = np.array([255., 255., 255.]) # Out of bounds
                else:
                    color = render_data[grid[i][j]]
                    obs[obs_i][obs_j] = color

        if self.orientation == 'right':
            obs = np.rot90(obs, k=1)
        elif self.orientation == 'down':
            obs = np.rot90(obs, k=2)
        elif self.orientation == 'left':
            obs = np.rot90(obs, k=3)

        return obs