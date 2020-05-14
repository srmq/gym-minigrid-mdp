"""
 This file is part of gym-minigrid-mdp.

 gym-minigrid-mdp is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 gym-minigrid-mdp is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with gym-minigrid-mdp.  If not, see <https://www.gnu.org/licenses/>. 

 Author: Sergio Queiroz <srmq@srmq.org>
"""

import numpy as np

from gym_minigrid.minigrid import (
    WorldObj, fill_coords, point_in_circle, COLORS, Grid,
    Goal, Lava, TILE_PIXELS    
)
from gym_minigrid.register import register
from gym_minigrid.window import Window
import gym
from gym import spaces
from gym.utils import seeding
from enum import IntEnum

class Agent(WorldObj):
    def __init__(self, color='purple'):
        super(Agent, self).__init__('agent', color)

    def can_pickup(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class AIMAGrid(Grid):
    def __init__(self):
        super().__init__(width=6, height=5)
        # Generate the surrounding walls
        self.wall_rect(0, 0, 6, 5)

        # Generate the wall at (2, 2)
        self.wall_rect(2, 2, 1, 1)
    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        #do not render the agent as a triangle
        img = super().render(tile_size, agent_pos, None, highlight_mask)
        return img


class AIMAEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Walk left, right, up, down
        left = 0
        right = 1
        up = 2
        down = 3

    class GoDirection(IntEnum):
        go_fwd = 0
        go_left = 1
        go_right = 2

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def remove_obj(self, i, j):
        ret = self.grid.get(i, j)
        self.grid.set(i, j, None)
        return ret

    def __init__(
        self,
        seed=1337,
        width=6, 
        height=5
    ):
        assert width != None and height != None
        self.width = width
        self.height = height

        # Range of possible rewards
        self.reward_range = (-1, 1)
        self.observation_space = spaces.Discrete(width*height)
        self.actions = AIMAEnv.Actions
        
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Book grid
        self.grid = AIMAGrid()

        # Window to use for human rendering mode
        self.window = None
        self.agent_start_pos = (1,3)
        self.agent_start_dir = 0

        # Initialize the RNG
        self.seed(seed=seed)

        self.mission = "get to the green goal square"

        self.reset()

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]


    def reset(self):        
        if hasattr(self, 'agent_pos') and self.agent_pos is not None:
            self.remove_obj(*self.agent_pos)
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, 1)

        # Place the lava at (4,2)
        self.put_obj(Lava(), 4, 2)
        
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        # Step count since episode start
        self.step_count = 0

        self.put_obj(Agent(), *self.agent_pos)
        obs = self.pos_to_obs(self.agent_pos)
        return obs

    @property
    def up_pos(self):
        return self.agent_pos + np.array((0, -1))

    @property
    def down_pos(self):
        return self.agent_pos + np.array((0, +1))

    @property
    def left_pos(self):
        return self.agent_pos + np.array((-1, 0))

    @property
    def right_pos(self):
        return self.agent_pos + np.array((+1, 0))


    def render(self, mode='human', close=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid_mdp')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    def pos_to_obs(self, pos):
        return int(pos[0]*self.width + pos[1])

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        self.step_count += 1
        rnd_num = self.np_random.random_sample()
        if rnd_num < 0.1:
            go_pos = AIMAEnv.GoDirection.go_left
            print("Go to left of movement")
        elif rnd_num < 0.9:
            go_pos = AIMAEnv.GoDirection.go_fwd
            print("Go forward of movement")
        else:
            go_pos = AIMAEnv.GoDirection.go_right
            print("Go to right of movement")
        
        if action == self.actions.left:
            if go_pos == AIMAEnv.GoDirection.go_fwd:
                candidate_pos = self.left_pos
            elif go_pos == AIMAEnv.GoDirection.go_left:
                candidate_pos = self.down_pos
            elif go_pos == AIMAEnv.GoDirection.go_right:
                candidate_pos = self.up_pos
            else:
                raise Exception("Unknown go_pos: " + str(go_pos))
        elif action == self.actions.right:
            if go_pos == AIMAEnv.GoDirection.go_fwd:
                candidate_pos = self.right_pos
            elif go_pos == AIMAEnv.GoDirection.go_left:
                candidate_pos = self.up_pos
            elif go_pos == AIMAEnv.GoDirection.go_right:
                candidate_pos = self.down_pos
            else:
                raise Exception("Unknown go_pos: " + str(go_pos))
        elif action == self.actions.up:
            if go_pos == AIMAEnv.GoDirection.go_fwd:
                candidate_pos = self.up_pos
            elif go_pos == AIMAEnv.GoDirection.go_left:
                candidate_pos = self.left_pos
            elif go_pos == AIMAEnv.GoDirection.go_right:
                candidate_pos = self.right_pos
            else:
                raise Exception("Unknown go_pos: " + str(go_pos))
        elif action == self.actions.down:
            if go_pos == AIMAEnv.GoDirection.go_fwd:
                candidate_pos = self.down_pos
            elif go_pos == AIMAEnv.GoDirection.go_left:
                candidate_pos = self.right_pos
            elif go_pos == AIMAEnv.GoDirection.go_right:
                candidate_pos = self.left_pos
            else:
                raise Exception("Unknown go_pos: " + str(go_pos))
        else: 
            raise Exception("Unkown action: " + str(action))
        candidate_cell = self.grid.get(*candidate_pos)
        reward = 0
        if candidate_cell == None or candidate_cell.can_overlap():
            self.remove_obj(*self.agent_pos)
            self.agent_pos = candidate_pos
            self.put_obj(Agent(), *self.agent_pos)
            reward = -0.04
        if candidate_cell != None and candidate_cell.type == 'goal':
            self.remove_obj(*self.agent_pos)
            done = True
            reward = +1
        if candidate_cell != None and candidate_cell.type == 'lava':
            self.remove_obj(*self.agent_pos)
            done = True
            reward = -1
        
        obs = self.pos_to_obs(self.agent_pos)

        return obs, reward, done, {}

register(
    id='MiniGrid-MDP-AIMAGrid-v0',
    entry_point='envs:AIMAEnv'
)
