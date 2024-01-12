import numpy as np
from scipy import interpolate
import cv2
import gymnasium as gym
from gym import spaces
from gym.utils import seeding

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

class Shape(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, step_size_action=3e-3, state_dim=64, max_num_step=1000):
        # State and action info
        self.state_dim = state_dim
        self.max_val = 4; self.min_val = -4
        self.max_act = 1; self.min_act = -1
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        self.state = None

        # Step info
        self.max_num_step = max_num_step
        self.num_step = 0
        self.step_size_action = step_size_action
        self.seed()

        # Shape interpolation info
        self.xk, self.yk = np.mgrid[-1:1:8j, -1:1:8j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])
    
    # For probabilistic action. TBA.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.state += self.step_size_action *action
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
        area, peri = geometry_info(self.state, self.xk, self.yk, self.xg, self.yg)
        done = (area == 0 or peri == 0)
        if not done:
            reward = -peri/np.sqrt(area)
            done = self.num_step > self.max_num_step
        else:
            reward = -1e9
        self.num_step = self.num_step+1
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at(mode='random')
    
    def reset_at(self, mode='random'):
        self.num_step = 0
        width = int(np.sqrt(self.state_dim))
        self.state = np.ones((width, width))
        if mode=='hole':
            self.state[:, 2:6, 2:6] = -1
        elif mode=='random':
            self.state = np.random.rand(width, width)
        elif mode=='random_with_padding':
            # Random with zero padding
            self.state[1:(width-1), :(width-1)] = np.random.rand(width-2, width-1)
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