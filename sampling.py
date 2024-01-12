import random
from collections import namedtuple, deque
import numpy as np
import torch

# Constant clipping value
MAX_VAL = 10.0

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data memory.
Data = namedtuple('Data', ('q', 'dat'))
class DataMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=int(capacity))
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Data(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        self.memory = deque([], maxlen=int(self.capacity))
    
# Q memory.
Q = namedtuple('Q', ('q'))
class QMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=int(capacity))
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Q(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        self.memory = deque([], maxlen=int(self.capacity))