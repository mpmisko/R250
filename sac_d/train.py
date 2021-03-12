import os
import yaml
import argparse
from datetime import datetime
import torch

from sacd.env import make_pytorch_env, make_apple_env
from sacd.agent import SacdAgent, SharedSacdAgent

def evaluate(test_env, agent1, agent2, rendering=False):
    num_episodes = 0
    num_steps = 0
    total_return = 0.0
    num_eval_steps = 10000

    while num_steps <= num_eval_steps:
        state = test_env.reset()
        if num_steps + 500 >= num_eval_steps and rendering:
            test_env.render()

        state = torch.transpose(torch.tensor(state.copy()), 0, -1).byte()
        episode_steps = 0
        episode_return = 0.0
        done = [False]
        while (not done[0]):
            action = agent1.exploit(state)
            next_state, reward, done, _ = test_env.step(action)

            if num_steps + 500 >= num_eval_steps and rendering:
                test_env.render()

            next_state = torch.transpose(torch.tensor(next_state[0].copy()), 0, -1).byte()
            num_steps += 1
            episode_steps += 1
            episode_return += reward[0]
            state = next_state

        num_episodes += 1
        total_return += episode_return


    mean_return = total_return / num_episodes

    if mean_return > agent1.best_eval_score:
        agent1.best_eval_score = mean_return
        agent1.save_models(os.path.join(agent1.model_dir, 'best'))

    agent1.writer.add_scalar(
        'reward/test', mean_return, agent1.steps)
    print('-' * 60)
    print(f'return: {mean_return:<5.1f}')
    print('-' * 60)


def train_episode(env, agent1, agent2, episode_cnt):
    episode_return = 0.
    episode_steps = 0

    done = [False]
    state = env.reset()
    state = torch.transpose(torch.tensor(state.copy()), 0, -1).byte()

    while (not done[0]):
        if agent1.start_steps > 500 * episode_cnt:
            action = env.action_space.sample()
        else:
            action = agent1.explore(state)
                    
        next_state, reward, done, _ = env.step(action)
        next_state = torch.transpose(torch.tensor(next_state[0].copy()), 0, -1).byte()
        
    
        clipped_reward = max(min(reward[0], 1.0), -1.0)

        # To calculate efficiently, set priority=max_priority here.
        agent1.memory.append(state, action, clipped_reward, next_state, done[0])

        agent1.steps += 1
        episode_steps += 1
        episode_return += reward[0]
        state = next_state

        if agent1.is_update():
            agent1.learn()

        if agent1.steps % agent1.target_update_interval == 0:
            agent1.update_target()

    # We log running mean of training rewards.
    agent1.train_return.append(episode_return)
    agent1.rewards.append(episode_return)

    if agent1.episodes % agent1.log_interval == 0:
        agent1.writer.add_scalar(
            'reward/train', agent1.train_return.get(), agent1.steps)

    print(f'Episode: {episode_cnt:<4}  '
            f'Episode steps: {episode_steps:<4}  '
            f'Return: {episode_return:<5.1f}')

def run_train(env, agent1, agent2, num_episodes=200):
    for i in range(num_episodes):
        train_episode(env, agent1, agent2, i)

        if i % 12 == 0 and i > 0:
            evaluate(env, agent1, agent2, rendering=False)
            agent1.save_models(os.path.join(agent1.model_dir, 'final'))

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_apple_env(args)
    test_env = make_apple_env(args)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent1 = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)

    agent2 = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)

    run_train(env, agent1, agent2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('./config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid_size_x', type=int, default=12)
    parser.add_argument('--grid_size_y', type=int, default=12)
    parser.add_argument('--apple_count', type=int, default=20)
    parser.add_argument('--agent_count', type=int, default=2)
    parser.add_argument('--observation_size', type=int, default=10)
    parser.add_argument('--num_episodes', type=int, default=250)
    parser.add_argument('--exp_steps', type=int, default=500)

    args = parser.parse_args()
    run(args)
