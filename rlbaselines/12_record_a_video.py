import os

import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from rlbaselines.common_utils.utils import record_video, turn_off_log_warnings

turn_off_log_warnings()

env_id = 'CartPole-v1'
video_folder = './logs/videos/'
os.makedirs(video_folder, exist_ok=True)
video_length = 1000

env = DummyVecEnv([lambda: gym.make(env_id)])
model = PPO2('MlpPolicy', env, verbose=0).learn(total_timesteps=int(1e5), log_interval=10)

record_video(env, model, video_folder=video_folder, video_length=video_length, prefix='ppo2')
model.save('./model/gym/ppo2_cartpole_v1.pkl')
