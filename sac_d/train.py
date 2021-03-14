import os
import yaml
import argparse
from datetime import datetime
import torch

from sacd.env import make_pytorch_env, make_apple_env
from sacd.agent import SacdAgent, SharedSacdAgent
from sacd.agent.dqn_agent import DQNAgent

def evaluate(test_env, agent1, agent2, rendering=False):
    num_episodes = 0
    num_steps = 0
    total_return1 = 0.0
    total_return2 = 0.0
    num_eval_steps = 1500

    while num_steps <= num_eval_steps:
        states = test_env.reset()
        if num_steps + 500 >= num_eval_steps and rendering:
            test_env.render()

        episode_steps = 0
        episode_return1 = 0.0
        episode_return2 = 0.0
        dones = [False, False]
        while not all(dones):
            action1 = agent1.exploit(states[0])
            action2 = agent2.exploit(states[1])

            next_states, rewards, dones, _ = test_env.step([action1, action2])

            if num_steps + 500 >= num_eval_steps and rendering:
                test_env.render()

            num_steps += 1
            episode_steps += 1
            episode_return1 += rewards[0]
            episode_return2 += rewards[1]
            states = next_states

        num_episodes += 1
        total_return1 += episode_return1
        total_return2 += episode_return2

    mean_return1 = total_return1 / num_episodes
    mean_return2 = total_return2 / num_episodes

    print('-' * 60)
    print(f'Return (agent 1): {mean_return1:<5.1f},  Return (agent 2): {mean_return2:<5.1f}')
    print('-' * 60)


def train_episode(env, agent1, agent2, episode_cnt):
    episode_return1 = 0.
    episode_return2 = 0.

    episode_steps = 0

    dones = [False, False]
    states = env.reset()

    while not all(dones):
        action1 = agent1.select_action(states[0])
        action2 = agent2.select_action(states[1])
        
        a0 = action1.item() if type(agent1) == DQNAgent else action1
        a1 = action2.item() if type(agent2) == DQNAgent else action2

        next_states, rewards, dones, _ = env.step([a0, a1])
        
        n0 = None if type(agent1) == DQNAgent and dones[0] else next_states[0]
        n1 = None if type(agent2) == DQNAgent and dones[1] else next_states[1]

        agent1.memory.append(dones[0], states[0], action1, n0, max(min(rewards[0], 1.0), -1.0)) # Fix this for DQN (expects none)
        agent2.memory.append(dones[1], states[1], action2, n1, max(min(rewards[1], 1.0), -1.0))

        agent1.steps += 1
        agent2.steps += 1
        episode_steps += 1
        episode_return1 += rewards[0] / 10
        episode_return2 += rewards[1] / 10

        states = next_states

        if agent1.is_update():
            agent1.learn()

        if agent1.steps % agent1.target_update_interval == 0:
            agent1.update_target()

        if agent2.is_update():
            agent2.learn()

        if agent2.steps % agent2.target_update_interval == 0:
            agent2.update_target()

    # We log running mean of training rewards.
    agent1.train_returns.append(episode_return1)
    agent2.train_returns.append(episode_return2)

    print(f'Episode: {episode_cnt:<4}  '
            f'Episode steps: {episode_steps:<4}  '
            f'Return1: {episode_return1:<5.1f}  '
            f'Return2: {episode_return2:<5.1f}  ')

    return agent1.train_returns, agent2.train_returns

def get_agents(env, test_env, log_dir, sacd_config, dqn_config, hyperparams):
    if args.mode=='sacd':
        if hyperparams:
            sacd_config.update(hyperparams)

        agent1 = SacdAgent(
            env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
            seed=args.seed, **sacd_config)

        agent2 = SacdAgent(
            env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
            seed=args.seed, **sacd_config)

    elif args.mode=='dqn':
        if hyperparams:
            dqn_config.update(hyperparams) 

        print(dqn_config)
        agent1 = DQNAgent(
            env=env, test_env=test_env, cuda=args.cuda, **dqn_config)

        agent2 = DQNAgent(
            env=env, test_env=test_env, cuda=args.cuda, **dqn_config)

    elif args.mode=='mixed':
        agent1 = SacdAgent(
            env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
            seed=args.seed, **sacd_config)

        agent2 = DQNAgent(
            env=env, test_env=test_env, cuda=args.cuda, **dqn_config)
    else:
        raise RuntimeError

    return agent1, agent2

def run_train(env, agent1, agent2, num_episodes=200):
    for i in range(num_episodes):
        train_episode(env, agent1, agent2, i)


def run_sacd_hyperparam_eval(env, num_episodes, sacd_config, log_dir, path='./ablation_logs_sacd_extra.csv'):
    hyperparams = {
        'memory_size' : [20000],
        'target_entropy_ratio' : [0.95, 0.99],
        'use_per' : [True, False],
        'target_update_interval' : [400, 4000]
    }
    
    # Create logging file
    with open(path, 'w+') as f: 
        header = ''.join([f"{k}," for k in hyperparams]) + 'episode,' + 'agent_id,' + 'reward\n'
        f.write(header)

    for memory_size in hyperparams['memory_size']:
        for ratio in hyperparams['target_entropy_ratio']:
            for per in hyperparams['use_per']:
                for update_interval in hyperparams['target_update_interval']:
                    params = {
                        'memory_size' : memory_size,
                        'target_entropy_ratio' : ratio,
                        'use_per' : per,
                        'target_update_interval': update_interval
                    }

                    agent1, agent2 = get_agents(env, env, log_dir, sacd_config, sacd_config, params)
                    print(f"Starting traning hyperparam config: [memory] -- {memory_size}, [target_update_interval] -- {update_interval}, [target_entropy_ratio] -- {ratio}, [use_per] -- {int(per)}")
                    run_train(env, agent1, agent2, num_episodes)


                    # Log agent 1
                    with open(path, 'a') as f: 
                        for episode, val in enumerate(agent1.train_returns):
                            data = f"{memory_size},{ratio},{int(per)},{update_interval},{episode+1},{1},{val}\n"
                            f.write(data)

                    # Log agent 2
                    with open(path, 'a') as f: 
                        for episode, val in enumerate(agent2.train_returns):
                            data = f"{memory_size},{ratio},{int(per)},{update_interval},{episode+1},{2},{val}\n"
                            f.write(data)




def run_dqn_hyperparam_eval(env, num_episodes, dqn_config, log_dir, path='./ablation_logs.csv'):
    hyperparams = {
        'memory_size' : [200, 2000, 20000],
        'gamma' : [0.9, 0.99],
        'eps_decay' : [10, 100, 1000],
    }
    
    # Create logging file
    with open(path, 'w+') as f: 
        header = ''.join([f"{k}," for k in hyperparams]) + 'episode,' + 'agent_id,' + 'reward\n'
        f.write(header)

    for memory_size in hyperparams['memory_size']:
        for gamma in hyperparams['gamma']:
            for decay in hyperparams['eps_decay']:
                    params = {
                        'memory_size' : memory_size,
                        'gamma' : gamma,
                        'eps_decay' : decay,
                    }

                    agent1, agent2 = get_agents(env, env, log_dir, dqn_config, dqn_config, params)
                    print(f"Starting traning hyperparam config: [memory] -- {memory_size}, [gamma] -- {gamma}, [decay] -- {decay}")
                    run_train(env, agent1, agent2, num_episodes)


                    # Log agent 1
                    with open(path, 'a') as f: 
                        for episode, val in enumerate(agent1.train_returns):
                            data = f"{memory_size},{gamma},{decay},{episode+1},{1},{val}\n"
                            f.write(data)

                    # Log agent 2
                    with open(path, 'a') as f: 
                        for episode, val in enumerate(agent2.train_returns):
                            data = f"{memory_size},{gamma},{decay},{episode+1},{2},{val}\n"
                            f.write(data)


def run(args):
    with open(args.sacd_config) as f:
        sacd_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    with open(args.dqn_config) as f:
        dqn_confing = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_apple_env(args)
    test_env = make_apple_env(args)

    # Specify the directory to log.
    name = args.sacd_config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    if args.mode == 'dqn':
        run_dqn_hyperparam_eval(env, args.num_episodes, dqn_confing, log_dir)
    else:
        run_sacd_hyperparam_eval(env, args.num_episodes, sacd_config, log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sacd_config', type=str, default=os.path.join('./config', 'sacd.yaml'))
    parser.add_argument(
        '--dqn_config', type=str, default=os.path.join('./config', 'dqn.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--mode', type=str, default="sacd")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid_size_x', type=int, default=12)
    parser.add_argument('--grid_size_y', type=int, default=12)
    parser.add_argument('--apple_count', type=int, default=12)
    parser.add_argument('--agent_count', type=int, default=2)
    parser.add_argument('--observation_size', type=int, default=12)
    parser.add_argument('--num_episodes', type=int, default=80)
    parser.add_argument('--exp_steps', type=int, default=500)

    args = parser.parse_args()
    run(args)
