import numpy as np
from common_nets import Mlp
from hamiltonian_nets import HDNet, DerivativeHDNet
from utils import *
from sampling import *

MAX_CLIP_VAL = 1e2

def optimize_net(memory, net, net_index, optim, criterion, 
                 num_iter, batch_size=32, log_interval=100):
    total_loss = 0
    cnt = 0
    # Training.
    for i in range(num_iter):
        data = memory.sample(batch_size)
        batch = Data(*zip(*data))
        q = torch.cat([torch.Tensor(qi) for qi in batch.q]).to(DEVICE)
        
        # Loss calculation and update by backward.
        val_predicted = net(q)
        if net_index == -1:
            val_expected = torch.cat([torch.Tensor(d) for d in batch.dat]).to(DEVICE)
        else:
            val_expected = torch.cat([torch.Tensor(d[:, net_index:(net_index+1)]) for d in batch.dat]).to(DEVICE)
        loss = criterion(val_predicted, val_expected)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), MAX_CLIP_VAL)
        optim.step()
        total_loss += loss.item()

        # Logging.
        cnt += 1
        if (i+1) % log_interval == 0:
            print('Iter {}: loss {:.3f}.'.format(i+1, total_loss/cnt))
            cnt = 0
            total_loss = 0

def sample_by_simulate(env, memory, q, HDnet, num_step_forward, step_size=1e-3, sample_rate=10, net_type='objective'):
    p = np.zeros((q.shape[0], q.shape[1]))
    samples = np.zeros((q.shape[0], num_step_forward, q.shape[1]))
    for i in range(num_step_forward):
        t = torch.tensor(i, dtype=torch.float).to(DEVICE)
        q_tensor = torch.tensor(q, dtype=torch.float).to(DEVICE)
        p_tensor = torch.tensor(p, dtype=torch.float).to(DEVICE)
        dp_tensor, mdq_tensor = torch.chunk(HDnet(t, torch.cat((q_tensor, p_tensor), axis=1)), 2, dim=1)
        q = q + step_size * dp_tensor.detach().cpu().numpy()
        p = p + step_size * mdq_tensor.detach().cpu().numpy()
        samples[:, i, :] = q #* (1 + 0.01*(np.random.rand(q.shape[0], q.shape[1])-.5))

    recorded_obj = [[] for _ in range(q.shape[0])]
    for j in range(q.shape[0]):
        for i in range(num_step_forward):
            cur_obj = env.obj(samples[j:(j+1), i, :])[0]
            if i % sample_rate == 0:
                if net_type == 'derivative':
                    memory.push(samples[j:(j+1), i, :], env.derivative(samples[j:(j+1), i, :]))
                elif net_type == 'objective':
                    memory.push(samples[j:(j+1), i, :], env.obj_comps(samples[j:(j+1), i, :]))
            recorded_obj[j].append(cur_obj)

    # Track objective trained and experience memory size.
    recorded_obj = np.array(recorded_obj)
    print(np.mean(recorded_obj[:,0:-1:sample_rate], axis=0))
    print('Memory size:', len(memory))

def train_hnet_by_objective(env_name, rate, mem_capacity, num_warmup_sample,\
        sample_size, sample_num_step_arr, sample_step_size, sample_rate,\
        num_iter_arr, lr=1e-3, batch_size=32, log_interval=100):
    # Setup environment
    env = get_environment(env_name)
    q_dim = env.q_dim
    
    # Setup Hamiltonian dynamics net and experience memory.
    obj_dims_arr, num_net, _ = get_architectures(env_name)
    nets = []
    for i in range(num_net):
        nets.append(Mlp(input_dim=q_dim, output_dim=1, 
                        layer_dims=obj_dims_arr[i], activation='relu').to(DEVICE))
    HDnet = HDNet(env_name, nets, rate).to(DEVICE)
    memory = DataMemory(capacity=mem_capacity)

    # Setup optimizers
    optims = []
    for i in range(num_net):
        optims.append(torch.optim.Adam(nets[i].parameters(), lr=lr))
    criterion = torch.nn.L1Loss()
    
    # Warm up first step.
    q_start = env.sample_q(num_warmup_sample, mode='random')
    for k in range(q_start.shape[0]):
        memory.push(q_start[k:(k+1)], env.obj_comps(q_start[k:(k+1)]))
    # q_memory.push(q_start[i:(i+1)])
    for i in range(num_net):
        print('\nWarming up net ' + str(i) + '...')
        optimize_net(memory, nets[i], i, optims[i], criterion, num_iter_arr[0], batch_size, log_interval)
    print('\nDone warming up step.\n\n')

    # Sample while optimize for the next several steps.
    for k in range(1, len(num_iter_arr)):
        print('\nSampling while optimizing in iter {}: '.format(k))
        # q_next = q_memory.sample(sample_size)
        # sample_data = memory.sample(sample_size)
        # sample_data = Data(*zip(*sample_data))
        # q = np.concatenate([qi for qi in sample_data.q], axis=0)
        q = env.sample_q(sample_size, mode='random')
        sample_by_simulate(env, memory, q, HDnet, num_step_forward=sample_num_step_arr[k], 
                           step_size=sample_step_size, sample_rate=sample_rate,
                           net_type='objective')
        for i in range(num_net):
            print('\nOptimzing net ' + str(i) + '...')
            optimize_net(memory, nets[i], i, optims[i], criterion, num_iter_arr[k], batch_size, log_interval)

    # Save objective fct net.
    for i in range(num_net):
        torch.save(nets[i].state_dict(), 'models/' + env_name + str(i) + '.pth')
    print('\nDone training for Hamiltonian net.')

def train_hnet_by_derivative(env_name, rate, mem_capacity, num_warmup_sample,\
        sample_size, sample_num_step_arr, sample_step_size, sample_rate,\
        num_iter_arr, lr=1e-3, batch_size=32, log_interval=100):
    # Setup environment
    env = get_environment(env_name)
    q_dim = env.q_dim
    
    # Setup Hamiltonian dynamics net and experience memory.
    _, _, derivative_dims_arr = get_architectures(env_name)
    derivative_net = Mlp(input_dim=q_dim, output_dim=q_dim, 
                        layer_dims=derivative_dims_arr, activation='relu').to(DEVICE)
    HDnet = DerivativeHDNet(derivative_net, rate).to(DEVICE)
    memory = DataMemory(capacity=mem_capacity)

    # Setup optimizers
    optim = torch.optim.Adam(derivative_net.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    
    # Warm up first step.
    q_start = env.sample_q(num_warmup_sample, mode='random')
    for i in range(q_start.shape[0]):
        memory.push(q_start[i:(i+1)], env.derivative(q_start[i:(i+1)]))
        # q_memory.push(q_start[i:(i+1)])
    optimize_net(memory, derivative_net, -1, optim, criterion, num_iter_arr[0], batch_size, log_interval)
    print('Done warming up step.\n')

    # Sample while optimize for the next several steps.
    for k in range(1, len(num_iter_arr)):
        print('\nSampling while optimizing in iter {}: '.format(k))
        # q_next = q_memory.sample(sample_size)
        # sample_data = memory.sample(sample_size)
        # sample_data = Data(*zip(*sample_data))
        # q = np.concatenate([qi for qi in sample_data.q], axis=0)
        q = env.sample_q(sample_size, mode='random')
        sample_by_simulate(env, memory, q, HDnet, num_step_forward=sample_num_step_arr[k], 
                           step_size=sample_step_size, sample_rate=sample_rate,
                           net_type='derivative')
        optimize_net(memory, derivative_net, optim, criterion, num_iter_arr[k], batch_size, log_interval)

    # Save derivative fct net.
    torch.save(derivative_net.state_dict(), 'models/' + env_name + '_derivative.pth')
    print('\nDone training for Hamiltonian net.')

def train_hnet_default_args(env_name, sample_num_step_arr, num_iter_arr, net_type='objective'):
    rate, mem_capacity, num_warmup_sample,\
    sample_size, sample_step_size, sample_rate,\
    lr, batch_size, log_interval = get_train_params(env_name)

    if net_type == 'derivative':
        print('Derivative mode:')
        train_hnet_by_derivative(env_name, rate, mem_capacity, num_warmup_sample,\
            sample_size, sample_num_step_arr, sample_step_size, sample_rate,\
            num_iter_arr, lr, batch_size, log_interval)
    elif net_type == 'objective':
        print('Objective mode:')
        train_hnet_by_objective(env_name, rate, mem_capacity, num_warmup_sample,\
            sample_size, sample_num_step_arr, sample_step_size, sample_rate,\
            num_iter_arr, lr, batch_size, log_interval)