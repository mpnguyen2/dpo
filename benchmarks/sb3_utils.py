import numpy as np
import pandas as pd
import torch
import pyrosetta
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
import imageio
from envs import Shape, ShapeBoundary, Molecule

DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print(DEVICE)

def get_environment(env_name):
    if env_name == 'shape':
        return ShapeBoundary()
    if env_name == 'shape_boundary':
        return Shape()
    if env_name == 'molecule':
        pose = pyrosetta.pose_from_sequence('A'*8)
        return Molecule(pose=pose)

def toList(s):
    tokens = s[1:-1].split(",")
    ans = []
    for token in tokens:
        ans.append(int(token))
    return ans

def get_architecture(env_name, on_policy=True):
    # Get architecture info from arch_file
    df = pd.read_csv('arch.csv')
    net_info = df[df['env_name']==env_name]
    actor_dims = toList(net_info['actor_dims'].values[0])
    critic_dims_field = 'v_critic_dims' if on_policy else 'q_critic_dims'
    critic_dims = toList(net_info[critic_dims_field].values[0])
    return actor_dims, critic_dims

def train(RL_method, env_name, total_samples, common_dims=[], activation='ReLU', lr=3e-3):
    # On-policy mean optimize policy directly using current policy being optimized.
    on_policy = RL_method in set(['PPO', 'TRPO', 'A2C'])
    # off_policy = set('SAC', 'DDPG', 'TD3')

    # Construct environment
    env = get_environment(env_name)
    actor_dims, critic_dims = get_architecture(env_name, on_policy=on_policy)

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
    if RL_method == 'TRPO':
        model = TRPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    elif RL_method == 'PPO':
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    elif RL_method == 'SAC':
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.device = DEVICE

    # Train and save model with SB3.
    model.learn(total_timesteps=total_samples, log_interval=10)
    model_path ="models/" + env_name + '_' + RL_method
    model.save(model_path)

def load_model(RL_method, model_path):
    if RL_method == 'TRPO':
        model = TRPO.load(model_path)
    elif RL_method == 'PPO':
        model = PPO.load(model_path)
    elif RL_method == 'SAC':
        model = SAC.load(model_path)
    return model

def visualize(RL_method, env_name, num_step=100, extra_args='random'):
    env = get_environment(env_name)
    model_path = "models/" + env_name + '_' + RL_method
    model = load_model(RL_method, model_path)
    model.set_env(env)
    rewards = []; images = []
    num_iteration = 0
    obs = env.reset_at(mode=extra_args)
    for _ in range(num_step):
        images.append(img)
        action, _ = model.predict(obs)
        obs, reward, done ,_ = env.step(action)
        img = env.render(mode='rgb_array')
        rewards.append(reward)
        num_iteration += 1
        if done:
            break
    env.close()
    
    # Showing simulation result
    gif_name = 'results/' + env_name + '_' + RL_method
    imageio.mimsave(gif_name, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=20)

# Make num_traj run and average the obtained objective function.
def get_model_result(RL_method, env_name, num_traj, num_step_per_traj):
    env = get_environment(env_name)
    model_path = "models/" + env_name + '_' + RL_method
    model = load_model(RL_method, model_path)
    model.set_env(env)
    objs = [] # obj is minus the last reward of a trajectory   
    for _ in range(num_traj):
        obs = env.reset()
        num_iteration = 0
        for _ in range(num_step_per_traj):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            num_iteration += 1
            if done:
                break
        objs.append(-reward)

    return np.mean(np.array(objs))