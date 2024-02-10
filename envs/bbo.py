import numpy as np
import gymnasium as gym

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

class BBO(gym.Env):
    def __init__(self, naive, step_size, max_num_step, seed=42):
        # Naive: whether to proceed with usual reward or use PMP-based modified version.
        self.naive = naive
        
        # Discount info
        self.gamma = 0.99
        self.step_pow = 1.0
        self.gamma_inc = self.gamma**self.step_pow
        self.discount = 1.0

        # Step info
        self.max_num_step = max_num_step
        self.num_step = 0
        self.step_size = step_size

        # Create generator.
        self.rng = np.random.default_rng(seed=seed)

    def set_gamma(self, gamma):
        # Reset discount info
        self.gamma = gamma
        self.step_pow = 1.0
        self.gamma_inc = self.gamma**self.step_pow
        self.discount = 1.0

    def calculate_final_reward(self, val, action):
        if self.naive:
            reward = -val
        else:
            # Reward reshape
            self.discount *= self.gamma_inc
            reward = 1/(self.discount**2) * (np.sum(action**2)*0.5 - val)

        return reward
    
    def get_val(self, reward, action):
        if self.naive:
            return -reward 
        else:
            return np.sum(action**2)*0.5 - (reward * (self.discount**2))