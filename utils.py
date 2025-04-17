import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import torch
import pyrosetta
from envs import ShapeBoundary, Shape, Molecule
from common_nets import Mlp
from policy import Policy

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get correct environment
def get_environment(env_name):
    if env_name == 'naive_shape_boundary':
        return ShapeBoundary(naive=True)
    if env_name == 'shape_boundary':
        return ShapeBoundary()
    if env_name == 'naive_shape':
        return Shape(naive=True)
    if env_name == 'shape':
        return Shape()
    pose = pyrosetta.pose_from_sequence('A'*8)
    # ('TTCCPSIVARSNFNVCRLPGTSEAICATYTGCIIIPGATCPGDYAN')
    # pyrosetta.pose_from_pdb("molecule_files/1AB1.pdb") #pyrosetta.pose_from_sequence('A' * 10)
    if env_name == 'naive_molecule':
        return Molecule(pose=pose, naive=True)
    if env_name == 'molecule':
        return Molecule(pose=pose)

def from_str_to_2D_arr(s):
    tokens = s[2:-2].split("],[")
    ans = []
    for arr_str in tokens:
        arr = []
        elem_str = arr_str.split(",")
        for e in elem_str:
            arr.append(int(e))
        ans.append(arr)
    return ans

def str_to_list(s):
    tokens = s[1:-1].split(",")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

# Get neural net architecture
def get_architectures(env_name, zero_order, arch_file='arch.csv'):
    # Get architecture info from arch_file
    df = pd.read_csv(arch_file)
    net_info = df[df['env_name']==env_name]
    if zero_order:
        layer_dims = str_to_list(net_info['derivative_layer_dims'].values[0])
    else:
        layer_dims = str_to_list(net_info['val_layer_dims'].values[0])

    return layer_dims

# Get predefined training parameters from file for a specific environment. 
def get_train_params(env_name, param_file='params.csv'):
    # Get parameter info from param file
    df = pd.read_csv(param_file)
    info = df[df['env_name']==env_name]

    # Rate underlying hamiltonian dynamics formula.
    rate = float(info['rate'].values[0])

    # Number of trajectories per stage
    num_traj = int(info['num_traj'].values[0])
    
    # Step size for discretized ODE
    step_size = float(info['step_size'].values[0])
    
    # Optimization params: learning rate, batch size, how often logging
    # and number of optimization steps per each sampling stage.
    lr = float(info['lr'].values[0])
    batch_size = int(info['batch_size'].values[0])
    log_interval = int(info['log_interval'].values[0])

    return rate, num_traj, step_size, lr, batch_size, log_interval

def setup_main_net(env_name, zero_order, state_dim):
    layer_dims = get_architectures(env_name, zero_order)
    if zero_order:
        output_dim = 1
    else:
        output_dim = state_dim
    main_net = Mlp(input_dim=state_dim, output_dim=output_dim, 
                    layer_dims=layer_dims, activation='relu').to(DEVICE)
    
    return main_net


def setup_dpo_model(method, env, env_name):
    rate, _, step_size, _, _, _ = get_train_params(env_name)
    zero_order = method.endswith('zero_order')
    state_dim = env.state_dim
    main_net = setup_main_net(env_name, zero_order, state_dim)
    path = 'models/' + env_name + '_' + method + '.pth'
    #main_net.load_state_dict(torch.load(path))
    main_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    main_net.to(DEVICE)
    model = Policy(zero_order, main_net, rate, step_size)

    return model

### Plotting
def _bootstrap(data, n_boot=2000, ci=68):
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)

def _tsplot(ax, x, data, mode='bootstrap', **kw):
    est = np.mean(data, axis=0)
    if mode == 'bootstrap':
        cis = _bootstrap(data)
    else:
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
    p2 = ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    p1 = ax.plot(x, est, **kw)
    ax.margins(x=0)

    return p1, p2

def plot_eval_benchmarks(eval_dict, time_steps, title, mode='bootstrap', 
                         colors=['red', 'blue', 'green', 'orange'],
                         plot_dir='tmp.png'):
    methods = list(eval_dict.keys())
    ax = plt.gca()
    graphic_list = []
    for i, method in enumerate(methods):
        data = eval_dict[method]
        _, p2 = _tsplot(ax, np.array(time_steps), data, mode, label=method, color=colors[i])
        graphic_list.append(p2)
    ax.legend(graphic_list, methods)
    ax.set_title(title)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Evaluation cost')
    plt.savefig('output/' + plot_dir)
    plt.show()