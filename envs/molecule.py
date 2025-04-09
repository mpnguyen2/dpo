from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyrosetta
from pyrosetta import *
from pyrosetta.teaching import *
from envs.bbo import BBO

pyrosetta.init()
MAX_ABS = 1e18

### Generic continuous environment for reduced Hamiltonian dynamics framework
class Molecule(BBO):
    # 1e-2
    def __init__(self, pose, naive=False, reset_scale=1e-2, step_size=1e-1, max_num_step=6):
        # Superclass setup
        super(Molecule, self).__init__(naive, step_size, max_num_step)

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

        # PyMol visualization
        self.pmm = PyMOLMover()
        self.pmm.keep_history(True)
    
    def step(self, action):
        self.state += self.step_size * action
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        val = self.sfxn(self.pose)

        # Update number of step
        self.num_step += 1

        done = self.num_step >= self.max_num_step

        # Calculate final reward
        reward = self.calculate_final_reward(val, action)
        reward = np.clip(reward, -MAX_ABS, MAX_ABS)
        
        return np.array(self.state), reward, done, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        return self.reset_at(mode='random'), {}
        #return self.reset_at(mode='test'), {}
    
    def reset_at(self, mode='random'):
        if mode == 'random':
            self.state = self.reset_scale*(self.rng.random(self.state_dim)-.5)
        elif mode == 'test':
            not_ok = True
            while not_ok:
                self.discount = 1.0
                self.state = self.reset_scale*(self.rng.random(self.state_dim)-.5)
                for k in range(self.num_residue):
                    self.pose.set_phi(k+1, self.state[2*k]) 
                    self.pose.set_psi(k+1, self.state[2*k+1])
                val = self.sfxn(self.pose)
                not_ok = val < 1800
        return np.array(self.state)

    def render(self):
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        self.pmm.apply(self.pose)

        return None
     
    def close(self):
        pass