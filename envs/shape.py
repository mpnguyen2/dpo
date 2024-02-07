import numpy as np
from scipy import interpolate
import cv2
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

class Shape(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, naive=False, step_size=1e-2, state_dim=64, max_num_step=20):
        # Naive: whether to proceed with usual reward or use PMP-based modified version.
        self.naive = naive

        # State and action info
        self.state_dim = state_dim
        self.max_val = 4; self.min_val = -4
        self.max_act = 1; self.min_act = -1
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
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

        # Shape interpolation info
        self.xk, self.yk = np.mgrid[-1:1:8j, -1:1:8j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])

        # Create generator.
        self.rng = np.random.default_rng(seed=42)

        # Control regularize factor
        self.c = 1.0/16
 
    def step(self, action):
        self.state += self.step_size *action

        # Update number of step
        self.num_step += 1

        # Calculate value
        area, peri = geometry_info(self.state, self.xk, self.yk, self.xg, self.yg)
        done = (area == 0 or peri == 0)
        if not done:
            val = peri/np.sqrt(area)
            done = self.num_step >= self.max_num_step
        else:
            val = 1e9

        # Calculate final reward
        if self.naive:
            reward = -val
        else:
            # Reward reshape
            self.discount *= self.gamma_inc
            reward = 1/(self.discount**2) * (self.c * np.sum(action**2)*0.5 - val)

        return np.array(self.state), reward, done, False, {}

    def get_val(self, reward, action):
        if self.naive:
            return -reward 
        else:
            return self.c * np.sum(action**2)*0.5 - (reward * (self.discount**2))
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        return self.reset_at(mode='random'), {}
    
    def reset_at(self, mode='random'):
        self.num_step = 0
        width = int(np.sqrt(self.state_dim))
        self.state = np.ones((width, width))
        if mode=='hole':
            self.state[1:8, 1:8] = 0
            self.state += self.rng.random((width, width))
            self.state = np.clip(self.state, 0, 1)
        elif mode=='random':
            self.state = self.rng.random((width, width))
        elif mode=='random_with_padding':
            # Random with zero padding
            self.state[1:(width-1), :(width-1)] = self.rng.random((width-2, width-1))
        self.state -= .5
        self.state = self.state.reshape(-1)
        return np.array(self.state)
    
    def render(self):
        xk, yk, xg, yg = self.xk, self.yk, self.xg, self.yg
        return 255-spline_interp(self.state.reshape(xk.shape[0], yk.shape[0]), xk, yk, xg, yg)
    
    def close(self):
        if self.viewer:
            self.viewer = None

## Helper functions ##
# Spline interpolation for 2D density problem
def spline_interp(z, xk, yk, xg, yg):
    # Interpolate knots with bicubic spline
    tck = interpolate.bisplrep(xk, yk, z)
    # Evaluate bicubic spline on (fixed) grid
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    # Convert spline values to binary image
    C = 255/2; thresh = C
    img = np.array(zint*C+C).astype('uint8')
    # Thresholding give binary image, which gives better contour
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return thresh_img

def geometry_info_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    
    return area, peri

def geometry_info(z, xk, yk, xg, yg):
    img = spline_interp(z, xk, yk, xg, yg)
    return geometry_info_from_img(img)