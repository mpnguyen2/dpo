import numpy as np
from policy import Policy

class QuerySystem:
    def __init__(self, env):
        self.env = env
        self.step_size = self.env.step_size
        self.state_dim = env.state_dim
        self.policy = None
    
    def set_policy(self, policy: Policy):
        self.policy = policy
    
    def get_zero_order_info(self, max_step):
        self.initial_state, _ = self.env.reset()
        states = [self.initial_state]
        actions = []
        vals = []
        done = False
        action = np.zeros(states[-1].shape[0])
        for _ in range(max_step+1):
            if done:
                break
            action = self.policy.get_action(states[-1], action)
            state, reward, done, _, _ = self.env.step(action)
            val = self.env.get_val(reward, action)
            
            # Store results
            states.append(np.copy(state))
            actions.append(np.copy(action))
            vals.append(val)
        
        return states, actions, vals

    def get_first_order_info(self, max_step, noise, directions):
        states, actions, _ = self.get_zero_order_info(max_step)
        vals_over_directions = []
        for d in directions:
            vals_over_directions.append([])
            self.env.state = self.initial_state + noise * d
            for action in actions:          
                _, reward, _, _, _ = self.env.step(action)
                val = self.env.get_val(reward, action)
                vals_over_directions[-1].append(val)

        return states, actions, vals_over_directions
