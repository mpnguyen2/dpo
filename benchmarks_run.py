import warnings
warnings.filterwarnings("ignore")
import math, collections
import pandas as pd
from utils import get_environment, setup_ocf_model
from benchmarks.sb3_utils import setup_benchmark_model
from test import get_model_avg_final_vals

DEFAULT_GAMMA = 0.99

base_methods = ['OCF_zero_order'] #, 'OCF_first_order', 'TRPO', 'PPO', 'SAC']
env_names = ['shape_boundary', 'naive_shape_boundary'] #, 'shape', 'naive_shape'] 
#['shape_boundary', 'naive_shape_boundary', 'shape', 'naive_shape', 'molecule', 'naive_molecule']

# Test basic params
num_traj_dict = {
                'shape_boundary': 200,
                'naive_shape_boundary': 200,
                'shape': 200,
                'naive_shape': 200,
                'molecule': 200,
                'naive_molecule': 200
                }
num_step_per_traj_dict = {
                'shape_boundary': 15,
                'naive_shape_boundary': 20,
                'shape': 20,
                'naive_shape': 20,
                'molecule': 10,
                'naive_molecule': 10
                }

env_to_gammas_dict = {
    'shape_boundary': set([DEFAULT_GAMMA, 0.9, 0.8]),
    'shape': set([DEFAULT_GAMMA, 0.9, 0.8, 0.6]),
    'molecule': set([DEFAULT_GAMMA, 0.9, 0.8, 0.6])
}

all_gammas = [DEFAULT_GAMMA, 0.9, 0.8, 0.6]
methods_with_gammas = []
for base_method in base_methods:
    if base_method.startswith('OCF'):
        methods_with_gammas.append((base_method, -1))
    else:
        for gamma in all_gammas:
            methods_with_gammas.append((base_method + '_' + str(gamma), gamma))

print('\n\n\n\n')
methods = [m for m, _ in methods_with_gammas]
print('Recording algorithms performance on trained models...')
print('Methods include:')
for i, m in enumerate(methods):
    print(str(i+1) + '. ' + m)

result = collections.defaultdict(list)
for env_name in env_names:
    env = get_environment(env_name)
    if env_name.startswith('naive'):
        avail_gammas = [DEFAULT_GAMMA]
    else:
        avail_gammas = env_to_gammas_dict[env_name]
    for method, gamma in methods_with_gammas:
            benchmark_model = not method.startswith('OCF')
            if not benchmark_model:
                if env_name.startswith('naive'):
                    result[env_name].append(math.nan)
                    continue
                model = setup_ocf_model(env, env_name, method)
            else:
                if gamma not in avail_gammas:
                    result[env_name].append(math.nan)
                    continue
                else:
                    # Load and setup trained benchmark model
                    model = setup_benchmark_model(method, env, env_name)
            print('\n\n')
            print('Testing ' + method + ' on ' + env_name)
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
