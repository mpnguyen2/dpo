import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy:
    def __init__(self, zero_order, main_net, rate=4e-3, step_size=1e-2):
        self.zero_order = zero_order
        self.main_net = main_net
        self.rate = rate
        self.step_size = step_size

    def get_action(self, state, action):
        if self.zero_order:
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float, requires_grad=True).to(DEVICE)
            H_tensor = self.main_net(state_tensor)
            dH_tensor = torch.autograd.grad(H_tensor.sum(), state_tensor, create_graph=True)[0]
            dH = dH_tensor.detach().cpu().numpy().reshape(-1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(DEVICE)
                dH_tensor = self.main_net(state_tensor)
                dH = dH_tensor.detach().cpu().numpy().reshape(-1)

        return action + self.step_size*(self.rate*action - dH)
    
    def get_main_net_val(self, state):
        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(DEVICE)
        with torch.no_grad():
            val_tensor = self.main_net(state_tensor)
        return val_tensor.detach().cpu().numpy().reshape(-1)
