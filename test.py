import numpy as np
import imageio

# Run num_traj trajectories using policy given by model on env.
def test_model_through_vals(seeds, env, model, num_traj, num_step_per_traj,
                                 benchmark_model=False):
    # List of all vals across seed & trajectory of shape (seed, num_traj, num_step_per_traj).
    all_vals = [[] for _ in range(len(seeds))]
    for i, seed in enumerate(seeds):
        env.rng = np.random.default_rng(seed=seed)
        for _ in range(num_traj):
            obs, _ = env.reset() # obs same as state in our case.
            num_iteration = 0
            action = np.zeros(obs.shape)
            vals_cur_traj = []
            for _ in range(num_step_per_traj):
                if benchmark_model:
                    action, _ = model.predict(obs)
                else:
                    action = model.get_action(obs, action)
                obs, reward, done, _, _ = env.step(action)
                val = env.get_val(reward, action)
                vals_cur_traj.append(val)
                num_iteration += 1
                if done:
                    break
            all_vals[i].append(np.array(vals_cur_traj))
    all_vals = np.array(all_vals)

    return all_vals.reshape(-1, num_step_per_traj)

# Visualize a particular trajectories from given model's policy.
def visualize(env, model, num_step=100, benchmark_model=False,
              extra_args='random', img_path=None):
    vals = []; images = []
    num_iteration = 0
    obs = env.reset_at(mode=extra_args)
    action = np.zeros(obs.shape)
    for _ in range(num_step):
        if benchmark_model:
            action, _ = model.predict(obs)
        else:
            action = model.get_action(obs, action)
        obs, reward, done ,_, _ = env.step(action)
        img = env.render()
        images.append(img)
        vals.append(env.get_val(reward, action))
        num_iteration += 1
        if done:
            break
    env.close()
    
    # Save simulation.
    if img_path is not None:
        imageio.mimsave(img_path, [np.array(img) for img in images], format='wmv', fps=20)
