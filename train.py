import gym
from gym import envs
import argparse
from gym_env.gym_apple_grid.envs.apple_grid_env import AppleGridEnv
from dqn import train

parser = argparse.ArgumentParser()
parser.add_argument('--grid_size_x', type=int, default=12)
parser.add_argument('--grid_size_y', type=int, default=12)
parser.add_argument('--apple_count', type=int, default=20)
parser.add_argument('--agent_count', type=int, default=2)
parser.add_argument('--observation_size', type=int, default=10)
parser.add_argument('--num_episodes', type=int, default=250)
parser.add_argument('--exp_steps', type=int, default=500)

args = parser.parse_args()


env = AppleGridEnv()
env.init_env(dimensions=[args.grid_size_x, args.grid_size_y],
             num_apples=args.apple_count,
             num_actors=args.agent_count,
             episode_steps=args.exp_steps,
             obs_window_size=args.observation_size)

train(env, args, is_rendering=False)
