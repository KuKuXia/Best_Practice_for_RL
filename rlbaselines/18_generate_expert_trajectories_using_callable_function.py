from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()

import os
import gym
from stable_baselines.gail import generate_expert_traj

log_dir = './model/expert_trajectories/'
os.makedirs(log_dir, exist_ok=True)

env = gym.make("CartPole-v1")


# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def dummy_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    return env.action_space.sample()


# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly
generate_expert_traj(dummy_expert, save_path='./model/expert_trajectories/dummy_expert_cartpole', env=env,
                     n_episodes=10)
print('Training Ended.')
