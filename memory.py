import random
from collections import deque
import numpy as np
import torch
from policy import Policy
from query_system import QuerySystem

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_samples_from_one_pool(memory, batch_size):
    if len(memory) > 0:
        return random.sample(memory, min(len(memory), batch_size))
    else:
        return []
        
class ReplayMemory(object):
    def __init__(self, query_system: QuerySystem, zero_order, 
                 capacity=5e5, new_data_capacity=256,
                 noise=1e-6):
        # Initialize memory.
        self.capacity = capacity
        self.new_data_capacity = new_data_capacity
        self.memory = deque([], maxlen=int(capacity))
        self.reinforce_samples = deque([], maxlen=int(new_data_capacity))
        self.new_samples = deque([], maxlen=int(new_data_capacity))
        
        # Set query system
        self.query_system = query_system
        self.sample_dim = self.query_system.env.state_dim
        self.zero_order = zero_order
        self.noise = noise

    def __len__(self):
        return len(self.memory)

    def set_policy(self, policy: Policy):
        self.policy = policy
        self.query_system.set_policy(policy)

    # Return labeled samples (state, val) obtained from the query system.
    def _query_samples(self, max_step):
        samples = []
        if self.zero_order:
            states, _, vals = self.query_system.get_zero_order_info(max_step)
            for state, val in zip(states[:-1], vals):
                samples.append((state, np.array([val])))
        else:
            n = self.sample_dim
            id_mat = np.eye(n)
            directions = [id_mat[i] for i in range(n)] + [-id_mat[i] for i in range(n)]
            states, _, vals_over_directions =\
                self.query_system.get_first_order_info(max_step, self.noise, directions)
            num_step = len(states)-1
            for i in range(num_step):
                grad = np.zeros(n)
                for d in range(n):
                    grad[d] = (vals_over_directions[d][i] - vals_over_directions[d+n][i])/ (2*self.noise)
                samples.append((states[i], grad))
        
        return samples

    def add_samples(self, num_traj, max_step, k):
        for _ in range(num_traj):
            samples = self._query_samples(max_step)

            # During first k step, store the label given by previous policy.
            for i in range(k):
                state, _ = samples[i]
                prev_policy_val = self.policy.get_main_net_val(state)
                self.reinforce_samples.append((state, prev_policy_val))
            
            # For the remaining step, store the label given by environment
            # on trajectories sampled from previous policy.
            self.new_samples.extend(samples[k:])

    def get_samples(self, batch_size):
        # Sample from main pool only (for now).
        samples = _get_samples_from_one_pool(self.memory, batch_size)
        #get_samples_from_one_pool(self.new_samples, batch_size) 
        #if reinforce:
        #samples +=_get_samples_from_one_pool(self.reinforce_samples, batch_size)

        return samples

    def refresh(self):
        # Push and new samples from current stage to main memory.
        self.memory.extend(self.reinforce_samples)
        self.memory.extend(self.new_samples)
        self.reinforce_samples.clear()
        self.new_samples.clear()
