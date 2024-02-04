import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame import gfxdraw

MAX_ACT = 1e4

class ShapeBoundary(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(self, naive=False, step_size=1e-2, state_dim=16, max_num_step=100):
        # Naive: whether to proceed with usual reward or use PMP-based modified version.
        self.naive = naive

        # State and action info
        self.state_dim = state_dim
        self.min_val = -4; self.max_val = 4
        self.observation_space = spaces.box.Box(low=self.min_val, high=self.max_val, shape=(state_dim,), dtype=np.float32)
        self.min_act = -1; self.max_act = 1
        self.action_space = spaces.box.Box(low=self.min_act, high=self.max_act, shape=(state_dim,), dtype=np.float32)
        self.state = None

        # Discount info
        self.gamma = 0.99
        self.step_pow = 0.1
        self.gamma_inc = self.gamma**self.step_pow
        self.discount = 1.0

        # Step info
        self.max_num_step = max_num_step
        self.num_step = 0
        self.step_size = step_size
    
        # Geometry
        self.num_coef = self.state_dim//2
        self.ts = np.linspace(0, 1, 80)
        self.verts = None

        # Rendering
        self.render_mode = 'human'
        self.screen_width = 600
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True

        # Create generator.
        self.rng = np.random.default_rng(seed=42)

    def step(self, action):
        self.state += self.step_size * action
        '''
        action_norm = np.max(np.abs(action))
        if action_norm > 0.5:
            self.state += 0.1*action
        elif action_norm > 0.1:
            self.state += 0.5*action
        elif action_norm < 0.01:
            action = self.np_random.rand(self.num_coef*2)
            self.state += action
        '''
        # Use cubic spline to smooth out the new state parametric curve
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        polygon = Polygon(zip(coords[:,0], coords[:,1]))
        coords = coords/np.max(np.abs(coords))*100 + 300
        self.verts = list(zip(coords[:,0], coords[:,1]))
        done = (polygon.area == 0 or polygon.length == 0)
        
        # Calculate value
        if not done:
            val = polygon.length/np.sqrt(polygon.area)
            done = self.num_step > self.max_num_step
        else:
            val = 1e9

        # Update number of step
        self.num_step += 1

        # Calculate final reward
        if self.naive:
            reward = -val
        else:
            # Reward reshape
            self.discount *= self.gamma_inc
            reward = 1/(self.discount**2) * (np.sum(action**2)*0.5 - val)
            
        return np.array(self.state), reward, done, False, {}

    def get_val(self, reward, action):
        if self.naive:
            return -reward 
        else:
            return np.sum(action**2)*0.5 - (reward * (self.discount**2))
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0

        return self.reset_at(mode='half_random'), {}
    
    def reset_at(self, mode='random'):
        self.num_step = 0
        self.state = np.zeros(self.state_dim)
        t = np.arange(self.num_coef)/self.num_coef
        if mode == 'ellipse':
            self.state[:self.num_coef] = 0.2*np.sin(2*np.pi*t)
            self.state[self.num_coef:] = np.cos(2*np.pi*t)
        elif mode == 'rect':
            # Assume n%4 == 0
            n = self.num_coef//4
            # x-coord
            self.state[:n] = np.arange(n)/n
            self.state[n:2*n] = 1
            self.state[2*n:3*n] = 1 - (np.arange(n)/n)
            self.state[3*n:4*n] = 0
            # y-coord
            self.state[4*n:5*n] = 0
            self.state[5*n:6*n] = np.arange(n)/n
            self.state[6*n:7*n] = 1
            self.state[7*n:8*n] = 1 - (np.arange(n)/n)
        elif mode == 'half_random':
            # Assume n%2 == 0
            n = self.num_coef//2
            # x-coord
            self.state[0:n] = 0.8*self.rng.random(n) + 0.2
            self.state[n:2*n] = -0.8*self.rng.random(n) - 0.2
            # y-coord
            self.state[2*n:3*n] = np.arange(n)/n
            self.state[3*n:4*n] = np.arange(n)/n
        elif mode == 'random':
            self.state = self.rng.random(self.state_dim) - 0.5
            
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        coords = coords/np.max(np.abs(coords))*100 + 300
        self.verts = sorted(list(zip(coords[:,0], coords[:,1])))
        return np.array(self.state)
    
    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        gfxdraw.aapolygon(self.surf, self.verts, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, self.verts, (0, 0, 0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
     
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False