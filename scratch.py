'''
# Closed form test for forward molecule.
EPS = 1e-5

def h(env, q, p, t, rate=1):
    return (-0.5*np.exp(-rate*t)*np.sum(p**2) - np.exp(rate*t)*env.obj(q.reshape(1, -1))[0])

def h_q(env, q, p, t, rate=1):
    n = q.shape[0]
    In = np.eye(n)
    ans = np.zeros(n)
    for i in range(n):
        ans[i] = (h(env, q+EPS*In[i], p, t, rate)-h(env, q-EPS*In[i], p, t, rate))/(2*EPS)
    return ans

def h_p(env, q, p, t, rate=1):
    n = q.shape[0]
    In = np.eye(n)
    ans = np.zeros(n)
    for i in range(n):
        ans[i] = (h(env, q, p+EPS*In[i], t, rate)-h(env, q, p-EPS*In[i], t, rate))/(2*EPS)
    return ans

def simulate_pmp(env_name, rate=1e-4, num_iter=1000, step_size=1e-2, log_interval=10, show_interval=10):
    env = get_environment(env_name)
    q = env.sample_q(1, mode='test').reshape(-1)
    p = np.zeros(q.shape[0])
    for i in range(num_iter):
        q = q + step_size * h_p(env, q, p, t=i, rate=rate)
        p = p - step_size * h_q(env, q, p, t=i, rate=rate)
        if i % log_interval == 0:
            print(i, env.obj(q.reshape(1, -1)))
            print(h_q(env, q, p, t=i, rate=rate))
        if env_name == 'molecule':
            if i % show_interval == 0:
                env.render(q)
'''