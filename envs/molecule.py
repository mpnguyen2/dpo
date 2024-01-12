import numpy as np
from gym.utils import seeding
from pyrosetta import *
from pyrosetta.teaching import *
from envs.continuous_env import ContinuousEnv

pyrosetta.init()

### Generic continuous environment for reduced Hamiltonian dynamics framework
class MoleculeEnv(ContinuousEnv):
    def __init__(self, pose):
        super().__init__(q_dim=1, num_comp=1)

        # Score function
        self.sfxn = get_fa_scorefxn()   
        #self.sfxn.set_weight(fa_atr, 1.0)
        #self.sfxn.set_weight(fa_rep, 1.0)

        # Other info
        self.pose = pose
        self.num_residue = pose.total_residue()
        self.q_dim = self.num_residue*2
        self.id = np.eye(self.q_dim)
        
        # PyMol visualization
        self.pmm = PyMOLMover()
        self.pmm.keep_history(True)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Terminal cost g
    def obj(self, q):
        costs = np.zeros(q.shape[0])
        num_sample = q.shape[0]
        for i in range(num_sample):
            for k in range(self.num_residue):
                self.pose.set_phi(k+1, q[i][2*k]) 
                self.pose.set_psi(k+1, q[i][2*k+1])
            costs[i] = self.sfxn(self.pose)
        return costs
    
    # Sampling state q
    def sample_q(self, num_examples, mode='random'):
        if mode == 'random':
            return 30*(np.random.rand(num_examples, self.q_dim)-.5)
        elif mode == 'zero':
            # Set both phi and psi equal 0
            return np.zeros((num_examples, self.q_dim))
        elif mode == 'near_zero':
            return 10*(np.random.rand(num_examples, self.q_dim)-.5)
      
    def render(self, q):
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, q[2*k]) 
            self.pose.set_psi(k+1, q[2*k+1])
        self.pmm.apply(self.pose)