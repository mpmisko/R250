from gym.envs.registration import register

register(
    id='apple_grid-v0',
    entry_point='gym_apple_grid.envs:AppleGridEnv',
)
