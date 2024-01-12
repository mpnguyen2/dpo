import numpy as np
from utils import *
from common_nets import Mlp
from hamiltonian_nets import HDNet, DerivativeHDNet

## Closed form test for forward molecule.
EPS = 1e-5

def h(env, q, p, t, rate=1):
    return (-0.5*np.exp(-rate*t)*np.sum(p**2) - np.exp(rate*t)*env.obj(q.reshape(1, -1))[0])

def h_q(env, q, p, t, rate=1):
    n = q.shape[0]
    In = np.eye(n)
    ans = np.zeros(n)
    for i in range(n):
        ans[i] = (h(env, q+EPS*In[i], p, t, rate)-h(env, q-EPS*In[i], p, t, rate))/(2*EPS)
    return ans

def h_p(env, q, p, t, rate=1):
    n = q.shape[0]
    In = np.eye(n)
    ans = np.zeros(n)
    for i in range(n):
        ans[i] = (h(env, q, p+EPS*In[i], t, rate)-h(env, q, p-EPS*In[i], t, rate))/(2*EPS)
    return ans

def simulate_pmp(env_name, rate=1e-4, num_iter=1000, step_size=1e-2, log_interval=10, show_interval=10):
    env = get_environment(env_name)
    q = env.sample_q(1, mode='test').reshape(-1)
    p = np.zeros(q.shape[0])
    for i in range(num_iter):
        q = q + step_size * h_p(env, q, p, t=i, rate=rate)
        p = p - step_size * h_q(env, q, p, t=i, rate=rate)
        if i % log_interval == 0:
            print(i, env.obj(q.reshape(1, -1)))
            print(h_q(env, q, p, t=i, rate=rate))
        if env_name == 'molecule':
            if i % show_interval == 0:
                env.render(q)

## Test hnet model
def test_hnet(env_name, mode, 
              num_traj=0, sample_mode='random', add_noise=False,
              rate=1, net_type='objective', model_path_ext='',
              num_iter=1000, step_size=1e-3, 
              log_interval=100, render_interval=-1):
    env = get_environment(env_name)

    # Load trained nets with given architectures.
    if net_type == 'objective':
        # Setup Hamiltonian dynamics net and experience memory.
        obj_dims_arr, num_net, _ = get_architectures(env_name)
        nets = []
        for i in range(num_net):
            nets.append(Mlp(input_dim=env.q_dim, output_dim=1, 
                            layer_dims=obj_dims_arr[i], activation='relu').to(DEVICE))
            nets[i].load_state_dict(
                torch.load('models/' + env_name + model_path_ext + str(i) + '.pth'))
        HDnet = HDNet(env_name, nets, rate).to(DEVICE)
    elif net_type == 'derivative':
        _, _, derivative_dims_arr = get_architectures(env_name)
        derivative_net = Mlp(input_dim=env.q_dim, output_dim=env.q_dim, 
                            layer_dims=derivative_dims_arr, activation='relu').to(DEVICE)
        derivative_net.load_state_dict(
            torch.load('models/' + env_name + model_path_ext + 'derivative.pth'))
        HDnet = DerivativeHDNet(derivative_net, rate).to(DEVICE)

    # Sample initial configuration and begin simulation based on trained HDnet.
    if mode == 'calculate_mean_reward':
        # Run num_traj random trajectories and calculate the average final reward.
        q = env.sample_q(num_traj, mode=sample_mode)
        p = np.zeros((num_traj, q.shape[1]))
        for i in range(num_iter):
            # t = torch.tensor(i*step_size, dtype=torch.float).to(DEVICE)
            t = torch.tensor(i, dtype=torch.float).to(DEVICE)
            q_tensor = torch.tensor(q, dtype=torch.float).to(DEVICE)
            p_tensor = torch.tensor(p, dtype=torch.float).to(DEVICE)
            dp_tensor, mdq_tensor = torch.chunk(HDnet(t, torch.cat((q_tensor, p_tensor), axis=1)), 2, dim=1)
            q = q + step_size * dp_tensor.detach().cpu().numpy()
            if add_noise:
                q *= (1 + 0.01*(np.random.rand(q.shape[0], q.shape[1])-.5))
            p = p + step_size * mdq_tensor.detach().cpu().numpy()
        return np.mean(env.obj(q))

    elif mode == 'single_traj':
        # Calculate (display) data for single traj.
        q = env.sample_q(1, mode=sample_mode)
        p = np.zeros((1, q.shape[1]))
        qs = []
        for i in range(num_iter):
            # t = torch.tensor(i*step_size, dtype=torch.float).to(DEVICE)
            t = torch.tensor(i, dtype=torch.float).to(DEVICE)
            q_tensor = torch.tensor(q, dtype=torch.float).to(DEVICE)
            p_tensor = torch.tensor(p, dtype=torch.float).to(DEVICE)
            dp_tensor, mdq_tensor = torch.chunk(HDnet(t, torch.cat((q_tensor, p_tensor), axis=1)), 2, dim=1)
            q = q + step_size * dp_tensor.detach().cpu().numpy()
            p = p + step_size * mdq_tensor.detach().cpu().numpy()
            if i % log_interval == 0:
                print(i, env.obj(q))
                # print((-mdq_tensor.detach().cpu().numpy()))
            if render_interval != -1 and i % render_interval == 0:
                env.render(q.reshape(-1))
            qs.append(q)

        # Save configurations along the optimal dynamics.
        output_dir='output/optimal_traj_numpy/' + env_name
        np.save(output_dir, np.array(qs, dtype=float))