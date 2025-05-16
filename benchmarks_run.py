import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import collections, time
from collections import defaultdict
import numpy as np
import pandas as pd
from utils import get_environment, setup_dpo_model, plot_eval_benchmarks
from benchmarks.sb3_utils import setup_benchmark_model
from test import test_model_through_vals


DEFAULT_GAMMA = 0.99
ALL_COLORS = ["red", "blue", "green", "orange", "purple", "cyan",
              "magenta", "brown", "pink", "olive", "teal", "gold", "navy"]
DEFAULT_GAMMA = 0.99  # for all straight-forward (S) methods


def run_benchmark(datasets, algorithms, seeds, env_to_gamma, 
                  num_traj_dict, num_step_per_traj_dict,
                  dataset_display_names, output_std,
                  metrics_filename='output/benchmarks.csv', 
                  plot_dir='output'):
    results = collections.defaultdict(list)
    eval_dict = collections.defaultdict(dict)
    print('Recording algorithms performance on trained models...')
    print('Methods include:')
    for i, m in enumerate(algorithms):
        print(str(i+1) + '. ' + m)

    # Generate the same initial state for all seeds
    obs_dict = defaultdict(lambda: defaultdict(list))  # obs_dict[env_name][seed] = list of initial obs

    for env_name in datasets:
        num_traj = num_traj_dict[env_name]
        for seed in seeds:
            env = get_environment(env_name)
            env.rng = np.random.default_rng(seed)
            initial_obs = []
            for _ in range(num_traj):
                obs, _ = env.reset()
                initial_obs.append(obs)
            obs_dict[env_name][seed] = initial_obs  # Save for later use

    # Evaluate each algorithm
    start_time = time.time()
    for algo in algorithms:
        print(f'\nProcessing: {algo}')
        is_dpo = algo.startswith('DPO')
        is_s_variant = algo.startswith('S-')
        base_algo = algo.replace('S-', '')

        for env_name in datasets:
            if is_dpo:
                gamma = env_to_gamma[env_name]
                env = get_environment(env_name)
                model = setup_dpo_model(algo, env, env_name)
            else:
                if is_s_variant:
                    gamma = DEFAULT_GAMMA
                    prefix = 'naive_'
                else:
                    gamma = env_to_gamma[env_name]
                    prefix = ''
                model_path = f"benchmarks/models/{prefix}{env_name}_{base_algo}_{str(gamma).replace('.', '_')}"
                env = get_environment(env_name)
                model = setup_benchmark_model(algo, env, model_path)

            print(f'Testing {algo} on {env_name} (gamma={gamma})')
            vals = test_model_through_vals(
                seeds=seeds,
                env=env,
                model=model,
                obs_dict=obs_dict[env_name],
                num_step_per_traj=num_step_per_traj_dict[env_name],
                benchmark_model=not is_dpo
            )

            # Aggregate results
            final_vals = vals[:, -1]
            final_vals = final_vals.reshape(len(seeds), num_traj_dict[env_name])
            avg_final_vals = np.mean(final_vals, axis=1)
            avg = np.mean(avg_final_vals)
            std = np.std(avg_final_vals)
            algo_name = 'DPO' if algo == 'DPO_zero_order' else algo
            if output_std:
                print(f'{algo_name} on {env_name} => {avg:.3f} ± {std:.3f}')
            else:
                print(f'{algo_name} on {env_name} => {avg:.2f}')
            eval_dict[env_name][algo_name] = vals
            results[algo_name].append((avg, std))

    # Save results to CSV.
    if output_std:
        df = pd.DataFrame({
            k: [f'{mean:.3f} ± {std:.3f}' for mean, std in v]
            for k, v in results.items()
        }, index=datasets).T
    else:
        df = pd.DataFrame({
            k: [f'{mean:.2f}' for mean, std in v]
            for k, v in results.items()
        }, index=datasets).T
    df.columns = [dataset_display_names[col] for col in df.columns]
    df.to_csv(metrics_filename)
    print('\nDone evaluation. Benchmarking metrics saved. Plotting...')

    # Plotting val along trajectories.
    time_steps_dict = {
        'surface': np.linspace(0, 1, num_step_per_traj_dict['surface']+1),
        'grid': np.linspace(0, 1, num_step_per_traj_dict['grid']+1),
        'molecule': np.linspace(0, 1, num_step_per_traj_dict['molecule']+1)
    }
    for env_name, display_name in dataset_display_names.items():
        if env_name != 'molecule':
            std_scale = 1.0
        else:
            # Magnifying variation in hard molecule exploration to see variation of each method
            std_scale = 3e3
        plot_eval_benchmarks(eval_dict[env_name],
                        time_steps=time_steps_dict[env_name],
                        title='Benchmarks on ' + display_name,
                        mode='bootstrap',
                        colors=ALL_COLORS,
                        plot_filename=os.path.join(plot_dir, 'benchmarks_' + env_name + '.png'),
                        std_scale=std_scale)
        
    # Time report
    time_taken_in_hours = (time.time()-start_time)/3600
    print(f'Done getting benchmarking output. Took {time_taken_in_hours:.3f} hours')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple_seeds', type=int, default=0, help='Use single seeds or multiple seeds')
    args = parser.parse_args()
    multiple_seeds = bool(args.multiple_seeds)
    output_std = multiple_seeds

    # Seeds
    if multiple_seeds:
        seeds = [42, 75, 105, 122, 137, 203, 381, 411, 437, 479]
        metrics_filename = 'output/benchmarks_stat_analysis.csv'
        plot_dir = 'output/stat_analysis'
    else:
        seeds = [42]
        metrics_filename = 'output/benchmarks.csv'
        plot_dir = 'output'

    # Datasets
    datasets = ['surface', 'grid', 'molecule']

    # Benchmark algorithms
    algorithms = ['DPO_zero_order',
                'TRPO', 'PPO', 'SAC', 'DDPG', 'CrossQ', 'TQC',
                'S-TRPO', 'S-PPO', 'S-SAC', 'S-DDPG', 'S-CrossQ', 'S-TQC']

    # Gamma values
    env_to_gamma = {
        'surface': 0.99,
        'grid': 0.81,
        'molecule': 0.0067
    }

    # Trajectory config
    num_traj_dict = {'surface': 200, 'grid': 200, 'molecule': 200}
    num_step_per_traj_dict = {'surface': 20, 'grid': 20, 'molecule': 6}

    # Dataset display names
    dataset_display_names = {
        'surface': 'Surface modeling',
        'grid': 'Grid-based modeling',
        'molecule': 'Molecular dynamics'
    }
    run_benchmark(datasets, algorithms, seeds, env_to_gamma, 
                  num_traj_dict, num_step_per_traj_dict,
                  dataset_display_names, output_std,
                  metrics_filename, plot_dir)
