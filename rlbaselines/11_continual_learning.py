"""
You can also move from learning on one environment to another for continual learning (PPO2 on DemonAttack-v0,
then transferred on SpaceInvaders-v0)
"""

from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_atari_env

if __name__ == '__main__':

    # There already exists an environment generator
    # that will make and wrap atari environments correctly
    env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=12, seed=0)

    model = PPO2('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    # Close the processes
    env.close()

    # The number of environments must be identical when changing environments
    env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=12, seed=0)

    # change env
    model.set_env(env)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()
