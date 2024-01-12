This is a quick summary of the associated code included in this supplementary material.
The codebase on Github is also available upon request.
There are 3 examples included in the experiment: 
1. Shape optimization with trivial topology.
2. Shape optimization with non-trivial topology.
3. Molecular energy minimization.

Here are the basic information of the files included:
1. The main training and testing logs of the OCF (optimal control gradient flow) algorithm for all 3 examples are included in main.ipynb.
2. The training and testing use functions from pmp_model.py and test.py.
3. The Hamiltonian net models are in hamiltonian_nets.py.
4. The models folder in the main directory contains all trained models for the OCF algorithm.
5. The detailed architecture for OCF training is included in the arch.csv.
6. All default hyperparameters for OCF training are in params.csv.
7. The benchmarks folder includes all of the benchmark trainings of reinforcement learning algorithms (TRPO, PPO, SAC) on 3 examples.
Its subfolder models included all trained models of the benchmarks on 3 examples.

All models are trained on the NVIDIA A100 40GB GPU. 
The total training time on both our model and benchmarks is about 12 days.
Thus, you may use the trained models to save time.