import warnings
warnings.filterwarnings("ignore")
from sb3_utils import train_benchmark_model

methods = ['TRPO', 'PPO', 'SAC']
env_names = ['shape_boundary', 'naive_shape_boundary', 'shape', 'naive_shape', 'molecule', 'naive_molecule']

# Train basic params
total_samples_dict = {
                'shape_boundary': 1e6,
                'naive_shape_boundary': 1e6,
                'shape': 1e6,
                'naive_shape': 1e6,
                'molecule': 1e6,
                'naive_molecule': 1e6
                }

print('\n\n\n\n')
print('Training benchmark models.')
print('Models include:', methods)
for env_name in env_names:
    for method in methods:
        print('\n\n')
        print('Training ' + method + ' on ' + env_name)
        train_benchmark_model(method, env_name, total_samples=int(total_samples_dict[env_name]), 
              lr=3e-4, log_interval=500)