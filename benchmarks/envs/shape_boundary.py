import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon
import gymnasium as gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

class ShapeBoundary(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size_action=3e-3, state_dim=16, max_num_step=1000):
        # State and action info
        self.state_dim = state_dim
        self.min_val = -4; self.max_val = 4
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(state_dim,), dtype=np.float32)
        self.min_act = -1; self.max_act = 1
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(state_dim,), dtype=np.float32)
        self.state = None

        # Step info
        self.max_num_step = max_num_step
        self.num_step = 0
        self.step_size_action = step_size_action
        self.seed()
    
        # Geometry
        self.num_coef = self.state_dim//2
        self.ts = np.linspace(0, 1, 80)
        self.verts = None

        # Viewer
        self.viewer = None
    
    # For probabilistic purpose (unique seed for the obj). To be used.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.state += self.step_size_action*action
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
        self.verts = zip(coords[:,0], coords[:,1])
        done = (polygon.area == 0 or polygon.length == 0) 
        if not done:
            reward = -polygon.length/np.sqrt(polygon.area)
            done = self.num_step > self.max_num_step
        else:
            reward = -1e9
        self.num_step = self.num_step+1
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at(mode='random')
    
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
            self.state[0:n] = 0.8*self.np_random.rand(n) + 0.2
            self.state[n:2*n] = -0.8*self.np_random.rand(n) - 0.2
            # y-coord
            self.state[2*n:3*n] = np.arange(n)/n
            self.state[3*n:4*n] = np.arange(n)/n
        elif mode == 'random':
            self.state = np.random.rand(self.state_dim) - 0.5
            
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        coords = coords/np.max(np.abs(coords))*100 + 300
        self.verts = zip(coords[:,0], coords[:,1])
        return np.array(self.state)
    
    def render(self, mode='human'):
        if self.viewer is None:
            screen_width = 600; screen_height = 600
            self.viewer = rendering.Viewer(screen_width, screen_height)
        self.viewer.draw_polygon(list(self.verts))
        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))
     
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None