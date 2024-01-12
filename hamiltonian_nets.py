import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

EPS = 1e-5

# Construct objective nets from components:
class ObjectiveNet(nn.Module):
    def __init__(self, env_name, nets):
        super(ObjectiveNet, self).__init__()
        self.nets = nets
        self.env_name = env_name
    
    def forward(self, q):
        if self.env_name.startswith('shape'):
            return self.nets[0](q)**2/self.nets[1](q)
        else:
            return self.nets[0](q)
    
# (Forward) Hamiltonian dynamics network using objective network.
class HDNet(nn.Module):
    def __init__(self, env_name, nets, rate=1):
        super(HDNet, self).__init__()
        self.obj_net = ObjectiveNet(env_name, nets)
        self.rate = rate
    
    # Copy parameter from another objective net.
    def copy_params(self, obj_net):
        self.obj_net.load_state_dict(obj_net.state_dict())

    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float, requires_grad=True)
            x = one * x
            q, p = torch.chunk(x, 2, dim=1)
            H = -0.5*torch.exp(-self.rate*t)*torch.sum(p**2, dim=1, keepdim=True) \
                - torch.exp(self.rate*t)* self.obj_net(q)
            dq, dp = torch.autograd.grad(H.sum(), (q, p), create_graph=True)

            return torch.cat((dp, -dq), dim=1)

# (Forward) Hamiltonian dynamics network using derivative network.
class DerivativeHDNet(nn.Module):
    def __init__(self, derivative_net, rate=1):
        super(DerivativeHDNet, self).__init__()
        self.derivative_net = derivative_net
        self.rate = rate
    
    # Copy parameter from another objective net.
    def copy_params(self, obj_net):
        self.obj_net.load_state_dict(obj_net.state_dict())

    def forward(self, t, x):
        with torch.no_grad():
            one = torch.tensor(1, dtype=torch.float, requires_grad=True)
            x = one * x
            q, p = torch.chunk(x, 2, dim=1)
            dp = -torch.exp(-self.rate*t)*p
            dq = -torch.exp(self.rate*t)*self.derivative_net(q)
    
            return torch.cat((dp, -dq), dim=1)