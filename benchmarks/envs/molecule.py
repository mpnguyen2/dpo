import numpy as np
import gymnasium as gym
from gym import spaces
from gym.utils import seeding
from pyrosetta import *
from pyrosetta.teaching import *

pyrosetta.init()

### Generic continuous environment for reduced Hamiltonian dynamics framework
class Molecule(gym.Env):
    def __init__(self, pose, step_size_action=1e-2, max_num_step=2000):
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

        # Step info
        self.max_num_step = max_num_step
        self.num_step = 0
        self.step_size_action = step_size_action
        self.seed()

        # PyMol visualization
        self.pmm = PyMOLMover()
        self.pmm.keep_history(True)    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        self.state += self.step_size_action*action
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        reward = -self.sfxn(self.pose)
        done = self.num_step > self.max_num_step
        self.num_step = self.num_step+1
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at(mode='zero')
    
    def reset_at(self, mode='random'):
        if mode == 'random':
            self.state = 30*(np.random.rand(self.state_dim)-.5)
        elif mode == 'zero':
            # Set both phi and psi equal 0
            self.state = np.zeros(self.state_dim)
        elif mode == 'zero_random':
            self.state = 10*(np.random.rand(self.state_dim)-.5)
        return np.array(self.state)

    def render(self, mode='human', close=True):
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        self.pmm.apply(self.pose)

        return None
     
    def close(self):
        pass