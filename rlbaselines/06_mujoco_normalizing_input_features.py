"""
只能通过命令行运行
"""
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/longxiajun/RL/RLBench/Best_Practice_for_RL/'])

import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from rlbaselines.common_utils.utils import turn_off_log_warnings

import mujoco_py
import os

mj_path, _ = mujoco_py.utils.discover_mujoco()

turn_off_log_warnings()
env = DummyVecEnv([lambda: gym.make('Reacher-v2')])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

model = PPO2("MlpPolicy", env).learn(total_timesteps=10000)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = './model/gym/'
model.save(log_dir + 'ppo_reacher')
model.save(os.path.join(log_dir, 'vec_normalize.pkl'))

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
