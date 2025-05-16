import numpy as np
import imageio


def test_model_through_vals(seeds, env, model, obs_dict, num_step_per_traj, benchmark_model=False):
    num_traj = len(next(iter(obs_dict.values())))  # get number of traj from any seed

    all_vals = []

    for seed in seeds:
        env.rng = np.random.default_rng(seed)
        seed_vals = []

        for obs in obs_dict[seed]:
            # Get initial observation cost
            env.reset()
            env.state = obs.copy()
            prev_action = np.zeros_like(obs)
            obs, reward, done, _, _ = env.step(prev_action)
            cur_vals = [env.get_val(reward, prev_action)]

            # Reset again and simulate trajectory
            env.reset()
            env.state = obs.copy()
            for _ in range(num_step_per_traj):
                if benchmark_model:
                    action, _ = model.predict(obs)
                else:
                    action = model.get_action(obs, prev_action)
                obs, reward, done, _, _ = env.step(action)
                val = env.get_val(reward, action)
                cur_vals.append(val)
                prev_action = action
                if done:
                    break
            seed_vals.append(np.array(cur_vals))

        all_vals.append(seed_vals)

    return np.array(all_vals).reshape(len(seeds) * num_traj, num_step_per_traj+1)


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
