import warnings
warnings.filterwarnings("ignore")
import math, collections
import pandas as pd
from utils import get_environment, setup_ocf_model
from benchmarks.sb3_utils import setup_benchmark_model
from test import get_model_avg_final_vals

methods = ['OCF_zero_order', 'OCF_first_order', 'SAC'] # ['TRPO', 'PPO', 'SAC']
env_names = ['shape_boundary', 'naive_shape_boundary'] #'molecule', 'naive_shape'] #, 'shape', 'molecule']

# Test basic params
num_traj_dict = {
                'shape_boundary': 100,
                'naive_shape_boundary': 100,
                'shape': 1000,
                'naive_shape': 1000,
                'molecule': 1000,
                'naive_molecule': 1000
                }
num_step_per_traj_dict = {
                'shape_boundary': 100,
                'naive_shape_boundary': 100,
                'shape': 100,
                'naive_shape': 100,
                'molecule': 100,
                'naive_molecule': 100
                }

print('\n\n\n\n')
print('Recording algorithms performance on trained models...')
result = collections.defaultdict(list)
for env_name in env_names:
    env = get_environment(env_name)
    for method in methods:
        print('\n\n')
        print('Testing ' + method + ' on ' + env_name)
        # Load and setup trained models
        benchmark_model = not method.startswith('OCF')
        if benchmark_model:
            model = setup_benchmark_model(method, env, env_name)
        else:
            if 'naive' in env_name:
                result[env_name].append(math.nan)
                continue
            model = setup_ocf_model(env, env_name, method)

        # Get average final vals on trajectories in test.
        avg_final_vals = get_model_avg_final_vals(env, model,
                                                  num_traj=int(num_traj_dict[env_name]),
                                                  num_step_per_traj=int(num_step_per_traj_dict[env_name]),
                                                  benchmark_model=benchmark_model)
        print(method + ': ' + str(avg_final_vals))
        result[env_name].append(avg_final_vals)

# Save benchmark results.
pd.DataFrame(result, index=methods).to_csv('benchmarks.csv')
print('Done benchmarking.')
