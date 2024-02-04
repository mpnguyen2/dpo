import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import pyrosetta
from pyrosetta import *
from pyrosetta.teaching import *

pyrosetta.init()

### Generic continuous environment for reduced Hamiltonian dynamics framework
class Molecule(gym.Env):
    def __init__(self, pose, naive=False, reset_scale=90, step_size=1e-2, max_num_step=100):
        # Naive: whether to proceed with usual reward or use PMP-based modified version.
        self.naive = naive

        # Molecule info
        self.pose = pose
        self.num_residue = pose.total_residue()
        self.sfxn = get_fa_scorefxn() # Score function

        # State and action info
        self.state_dim = self.num_residue*2
        self.min_val = -180; self.max_val = 180
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        self.min_act = -90; self.max_act = 90 
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)      
        self.state = None

        # Reset scale
        self.reset_scale = reset_scale

        # Discount info
        self.gamma = 0.99
        self.step_pow = 0.1
        self.gamma_inc = self.gamma**self.step_pow
        self.discount = 1.0

        # Step info
        self.max_num_step = max_num_step
        self.num_step = 0
        self.step_size = step_size

        # PyMol visualization
        self.pmm = PyMOLMover()
        self.pmm.keep_history(True)

        # Create generator.
        self.rng = np.random.default_rng(seed=42)
    
    def step(self, action):
        self.state += self.step_size * action
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        val = self.sfxn(self.pose)
        done = self.num_step > self.max_num_step

        # Update number of step
        self.num_step += 1

        # Calculate final reward
        if self.naive:
            reward = -val
        else:
            # Reward reshape
            self.discount *= self.gamma_inc
            reward = 1/(self.discount**2) * (np.mean(action**2)*0.5 - val)
        
        '''
        # Debug info
        if self.num_step == 1:
            print('Begin: val:', val, '; reward:', reward)
        if done:
            print('End: val:', val, '; reward:', reward)
        '''
        
        return np.array(self.state), reward, done, False, {}

    def get_val(self, reward, action):
        if self.naive:
            return -reward 
        else:
            return np.mean(action**2)*0.5 - (reward * (self.discount**2))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        return self.reset_at(mode='random'), {}
    
    def reset_at(self, mode='random'):
        if mode == 'random':
            self.state = self.reset_scale*(self.rng.random(self.state_dim)-.5)
        elif mode == 'zero':
            # Set both phi and psi equal 0
            self.state = np.zeros(self.state_dim)
        return np.array(self.state)

    def render(self):
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        self.pmm.apply(self.pose)

        return None
     
    def close(self):
        pass