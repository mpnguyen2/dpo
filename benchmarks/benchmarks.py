import warnings
warnings.filterwarnings("ignore")
from sb3_utils import *

RL_methods = ['TRPO', 'PPO', 'SAC']
env_names = ['shape_boundary', 'shape', 'molecule']
total_samples_dict = {
                'shape_boundary': 1e7,
                'shape': 1e7,
                'molecule': 1e7
                }
num_traj_dict = {
                'shape_boundary': 1000,
                'shape': 1000,
                'molecule': 1000
                }
num_step_per_traj_dict = {
                'shape_boundary': 500,
                'shape': 500,
                'molecule': 1000
                }

print('Training baseline models...')
for env_name in env_names:
    for RL_method in RL_methods:
        print('Training ' + RL_method + ' on ' + env_name)
        train(RL_method, env_name, total_samples=int(total_samples_dict[env_name]))

print('Recording RL algorithms performance...')
result = {}
for env_name in env_names:
    for RL_method in RL_methods:
        print('Testing ' + RL_method + ' on ' + env_name)
        avg_objective = get_model_result(RL_method, env_name, num_traj=int(num_traj_dict[env_name]), 
                            num_step_per_traj=int(num_step_per_traj_dict[env_name]))
        print(RL_method + ': ' + str(avg_objective))
        if env_name not in result:
            result[env_name] = []
        result[env_name].append(avg_objective)

pd.DataFrame(result, index=RL_methods).to_csv('rl_benchmarks.csv')
print('Done benchmarking.')