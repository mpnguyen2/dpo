import numpy as np
import pandas as pd
import cv2 
import torch

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get correct environment
def get_environment(env_name):
    if env_name == 'cartpole':
        from envs.classical_controls import CartPole
        return CartPole()
    if env_name == 'shape_boundary':
        from envs.shape import ShapeBoundary
        return ShapeBoundary()
    if env_name == 'shape':
        from envs.shape import Shape
        return Shape()
    if env_name == 'molecule':
        from envs.molecule import MoleculeEnv
        import pyrosetta
        pose = pyrosetta.pose_from_sequence('A'*8)
        # ('TTCCPSIVARSNFNVCRLPGTSEAICATYTGCIIIPGATCPGDYAN')
        # pyrosetta.pose_from_pdb("molecule_files/1AB1.pdb") #pyrosetta.pose_from_sequence('A' * 10)
        return MoleculeEnv(pose=pose)
    
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

def from_str_to_1D_arr(s):
    tokens = s[1:-1].split(",")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

# Get neural net architecture
def get_architectures(env_name, arch_file='arch.csv'):
    # Get architecture info from arch_file
    df = pd.read_csv(arch_file)
    net_info = df[df['env_name']==env_name]
    obj_dims_arr = from_str_to_2D_arr(net_info['objective_dims'].values[0])
    derivative_dims_arr = from_str_to_1D_arr(net_info['derivative_dims'].values[0])
    
    return obj_dims_arr, len(obj_dims_arr), derivative_dims_arr

# Get predefined training parameters from file for a specific environment. 
def get_train_params(env_name, param_file='params.csv'):
    # Get parameter info from param file
    df = pd.read_csv(param_file)
    info = df[df['env_name']==env_name]

    # Hnet parameters (similar to discounted rate concept)
    rate = float(info['hnet_rate'].values[0])

    # Sampling params starting with replay memory size & number of samples in warming up step.
    mem_capacity = int(info['mem_capacity'].values[0])
    num_warmup_sample = int(info['num_warmup_sample'].values[0])

    # Number of trajectories per sampling step
    sample_size = int(info['sample_size'].values[0])
    
    # Discrete step size for discrete trajectories sampled.
    sample_step_size = float(info['sample_step_size'].values[0])
    
    # Sampling rate, which determine subset of points to be sampled on each trajectory.
    sample_rate = int(info['sample_rate'].values[0])
    
    # Optimization params: learning rate, batch size, how often logging
    # and number of optimization steps per each sampling stage.
    lr = float(info['lr'].values[0])
    batch_size = int(info['batch_size'].values[0])
    log_interval = int(info['log_interval'].values[0])

    return rate, mem_capacity, num_warmup_sample,\
        sample_size, sample_step_size, sample_rate,\
        lr, batch_size, log_interval


# Show image from numpy q data.
def display(env_name, input_file='output/optimal_traj_numpy/', 
            output_file='output/videos/test.wmv'):
    # Initialize environment
    env = get_environment(env_name) 
    isColor = True
    if env_name == 'shape':
        isColor = False
    if env_name != 'shape':
        env.render(np.zeros(env.q_dim))
    input_file += env_name + '.npy'

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (env.viewer.width, env.viewer.height), isColor=isColor)

    # Load numpy file
    qs = np.load(input_file)
    
    # Write rendering image
    for i in range(qs.shape[0]):
        out.write(env.render(qs[i].reshape(-1)))
    
    # Release video
    out.release()
    env.close()
    print('\nDone displaying the optimal trajectory!')