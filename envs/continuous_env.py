import numpy as np
from gym.utils import seeding

### Generic continuous environment for reduced Hamiltonian dynamics framework
class ContinuousEnv():
    def __init__(self, q_dim=1, num_comp=1):
        self.q_dim = q_dim
        self.num_comp = num_comp
        self.eps = 1e-5
        self.id = np.eye(q_dim)
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Black box objective function.
    def obj(self, q):
        return np.zeros(q.shape[0])
    
    # Black box objective function components.
    def obj_comps(self, q):
        return np.zeros((q.shape[0], self.num_comp))
    
    # Numerical derivative of objective fct.
    def derivative(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.obj(q+self.eps*self.id[i])-self.obj(q-self.eps*self.id[i]))/(2*self.eps)
        return ret
    
    # Sampling state q
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train':
            return np.zeros((num_examples, self.q_dim))
        else:
            return np.ones((num_examples, self.q_dim))
    
    # Image rendering
    def render(self, q):
        return