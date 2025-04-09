import warnings
warnings.filterwarnings("ignore")
from sb3_utils import train_benchmark_model
import time


methods = ['TQC', 'CrossQ', 'TRPO', 'PPO', 'SAC']
env_names = ['shape_boundary', 'naive_shape_boundary', 'shape', 'naive_shape', 'molecule', 'naive_molecule']

DEFAULT_GAMMA = 0.99

# Train basic params
total_samples_dict = {
            'shape_boundary': 1e5,
            'naive_shape_boundary': 1e5,
            'shape': 1e5,
            'naive_shape': 1e5,
            'molecule': 5000,
            'naive_molecule': 5000
            }

env_to_gammas_dict = {
    'shape_boundary': [0.99],
    'shape': [0.81],
    'molecule': [0.0067],
}

print('\n\n\n\n')
print('Training benchmark models.')
print('Models include:', methods)
print('Environments include:', env_names)
start_time = time.time()
for env_name in env_names:
    if env_name.startswith('naive'):
        gammas = [DEFAULT_GAMMA]
    else:
        gammas = env_to_gammas_dict[env_name]
    for method in methods:
        for gamma in gammas:
            individual_start_time = time.time()
            print('\n\n')
            print('Training ' + method + ' on ' + env_name + ' with gamma ' + str(gamma).replace('.', '_'))
            train_benchmark_model(method, gamma, env_name, total_samples=int(total_samples_dict[env_name]), 
                lr=3e-4, log_interval=50)
            print('Training takes {:.3f} hours'.format(int(time.time()-individual_start_time)/3600))

print('\n\nTotal training takes {:.3f} hours'.format(int(time.time()-start_time)/3600))