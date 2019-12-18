from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()

from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

import os

log_dir = './model/expert_trajectories/'
os.makedirs(log_dir, exist_ok=True)

# 预训练一个DQN神经网络生成10个轨迹作为expert策略样本
model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
# Train a DQN agent for 1e5 timesteps and generate 10 trajectories
# data will be saved in a numpy archive named `expert_cartpole.npz`
generate_expert_traj(model, './model/expert_trajectories/expert_dqn_cartpole', n_timesteps=int(1e5), n_episodes=10)
print('Training Ended.')
