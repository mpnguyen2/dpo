import warnings
warnings.filterwarnings("ignore")
import math, collections, time
import numpy as np
import pandas as pd
from utils import get_environment, setup_dpo_model, plot_eval_benchmarks
from benchmarks.sb3_utils import setup_benchmark_model
from test import test_model_through_vals


DEFAULT_GAMMA = 0.99
ALL_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "pink", "olive", "teal", "gold", "navy"]


seeds = [42] #, 75, 105, 122, 137, 203, 381, 411, 437, 479]

# Datasets (columns)
datasets = ['molecule'] #['shape_boundary', 'shape', 'molecule']

# Algo (rows): 1 DPO, 5 benchmark, 5 naive benchmark
algorithms = ['DPO_zero_order'] #,
              #'TRPO', 'PPO', 'SAC', 'DDPG', 'CrossQ', 'TQC',
              #'S-TRPO', 'S-PPO', 'S-SAC', 'S-DDPG', 'S-CrossQ', 'S-TQC']

# Gamma values
env_to_gamma = {
    'shape_boundary': 0.99,
    'shape': 0.81,
    'molecule': 0.0067
}
DEFAULT_GAMMA = 0.99  # for all S- methods

# Traj config
num_traj_dict = {'shape': 200, 'shape_boundary': 200, 'molecule': 200}
num_step_per_traj_dict = {'shape': 20, 'shape_boundary': 20, 'molecule': 6}

results = collections.defaultdict(list)
eval_dict = collections.defaultdict(dict)

print('Recording algorithms performance on trained models...')
print('Methods include:')
for i, m in enumerate(algorithms):
    print(str(i+1) + '. ' + m)

start_time = time.time()
for algo in algorithms:
    print(f'\nProcessing: {algo}')
    is_dpo = algo.startswith('DPO')
    is_s_variant = algo.startswith('S-')
    base_algo = algo.replace('S-', '')

    for env_name in datasets:
        env = get_environment(env_name)

        if is_dpo:
            gamma = env_to_gamma[env_name]
            model = setup_dpo_model(algo, env, env_name)
        else:
            if is_s_variant:
                gamma = DEFAULT_GAMMA
                prefix = 'naive_'
            else:
                gamma = env_to_gamma[env_name]
                prefix = ''
            model_path = "benchmarks/models/" + prefix + env_name
            model_path += '_' + base_algo + '_' + str(gamma).replace('.', '_')
            model = setup_benchmark_model(algo, env, model_path)
        
        print(f'Testing {algo} on {env_name} (gamma={gamma})')
        vals = test_model_through_vals(
            seeds, env, model,
            num_traj=num_traj_dict[env_name],
            num_step_per_traj=num_step_per_traj_dict[env_name],
            benchmark_model=not is_dpo
        )

        # Report average last values
        final_vals = vals[:, -1]
        final_vals = final_vals.reshape(len(seeds), num_traj_dict[env_name])
        final_vals = np.mean(final_vals, axis=1)
        #print(np.mean(final_vals, axis=-1))
        #print(np.std(final_vals, axis=-1))
        avg_final_vals = np.mean(final_vals)
        std_final_vals = np.std(final_vals)
        algo_name = 'DPO' if algo == 'DPO_zero_order' else algo
        print(f'{algo_name} on {env_name} => {avg_final_vals:.3f} ± {std_final_vals:.3f}')
        eval_dict[env_name][algo_name] = vals
        results[algo_name].append((avg_final_vals, std_final_vals))

# Save results to CSV.
df = pd.DataFrame({
    k: [f'{mean:.3f} ± {std:.3f}' for mean, std in v]
    for k, v in results.items()
}, index=datasets).T
# df = pd.DataFrame(results, index=datasets).T
dataset_display_names = {
    'shape_boundary': 'Materials deformation',
    'shape': 'Topological materials deformation',
    'molecule': 'Molecular dynamics'
}
df.columns = [dataset_display_names[col] for col in df.columns]
df.to_csv('output/benchmarks.csv')
print('\nDone. Results saved to benchmarks.csv')

# Plotting val along trajectories.
time_steps_dict = {
    'shape_boundary': np.linspace(0, 1, num_step_per_traj_dict['shape_boundary']),
    'shape': np.linspace(0, 1, num_step_per_traj_dict['shape']),
    'molecule': np.linspace(0, 1, num_step_per_traj_dict['molecule'])
}
for env_name, display_name in dataset_display_names.items():
    plot_eval_benchmarks(eval_dict[env_name],
                         time_steps=time_steps_dict[env_name],
                         title='Benchmarks on ' + display_name,
                         mode='bootstrap',
                         colors=ALL_COLORS,
                         plot_dir='benchmarks_' + env_name + '.png')


time_taken_in_hours = (time.time()-start_time)/3600
print(f'Done getting benchmarking output. Took {time_taken_in_hours:.3f} hours')
