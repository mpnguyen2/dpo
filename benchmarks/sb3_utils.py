import os, sys
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO

# current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
# path to main repo (locally)
sys.path.append(os.path.dirname(cur_dir))
from utils import get_environment, str_to_list, DEVICE

def get_RL_nets_architectures(env_name, on_policy=True):
    # Get architecture info from arch_file
    df = pd.read_csv('arch.csv')
    net_info = df[df['env_name']==env_name]
    actor_dims = str_to_list(net_info['actor_dims'].values[0])
    critic_dims_field = 'v_critic_dims' if on_policy else 'q_critic_dims'
    critic_dims = str_to_list(net_info[critic_dims_field].values[0])
    return actor_dims, critic_dims

def train_benchmark_model(method, env_name, total_samples, common_dims=[], 
          activation='ReLU', lr=3e-3, log_interval=10):
    # On-policy mean optimize policy directly using current policy being optimized.
    on_policy = method in set(['PPO', 'TRPO', 'A2C'])
    # off_policy = set('SAC', 'DDPG', 'TD3')

    # Construct environment
    env = get_environment(env_name)
    actor_dims, critic_dims = get_RL_nets_architectures(env_name, on_policy=on_policy)

    # Net architecture for actor and critic networks
    if on_policy:
        net_arch_dict = dict(pi=actor_dims, vf=critic_dims)
    else:
        net_arch_dict = dict(pi=actor_dims, qf=critic_dims)
    
    # Add common processing nets from state to both actor & critic.
    if len(common_dims) != 0:
        net_arch = []
        for dim in common_dims:
            net_arch.append(dim)
        net_arch.append(net_arch_dict)
    else:
        net_arch = net_arch_dict
        
    # Set the policy args
    activation_fn = torch.nn.ReLU if activation == 'ReLU' else torch.nn.Tanh
    policy_kwargs = dict(activation_fn=activation_fn,
                         net_arch=net_arch)
    
    # Build the model using SB3.
    if method == 'TRPO':
        model = TRPO("MlpPolicy", env, learning_rate=lr, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'PPO':
        model = PPO("MlpPolicy", env, learning_rate=lr, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'SAC':
        model = SAC("MlpPolicy", env, learning_rate=lr, policy_kwargs=policy_kwargs, verbose=1)
    model.device = DEVICE

    # Train and save model with SB3.
    model.learn(total_timesteps=total_samples, log_interval=log_interval)
    model_path ="models/" + env_name + '_' + method
    model.save(model_path)

def _load_benchmark_model(method, model_path):
    if method == 'TRPO':
        model = TRPO.load(model_path)
    elif method == 'PPO':
        model = PPO.load(model_path)
    elif method == 'SAC':
        model = SAC.load(model_path)
    return model

def setup_benchmark_model(method, env, env_name):
    model_path = "benchmarks/models/" + env_name + '_' + method
    model = _load_benchmark_model(method, model_path)
    model.set_env(env)

    return model