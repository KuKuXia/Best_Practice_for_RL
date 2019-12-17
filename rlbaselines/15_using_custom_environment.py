import gym
import numpy as np
from gym import spaces
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

from rlbaselines.common_utils.env_checker import check_env
from rlbaselines.common_utils.utils import turn_off_log_warnings


class CustomEev(gym.Env):
    """
    Custom environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEev, self).__init__()
        # Define action and observation space, they must be gym.spaces objects
        # Example when using discrete actions
        self.action_space = spaces.Discrete(2)
        # Example for using image as input
        self.observation_space = spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8())

    def step(self, action):
        # define the environment
        observation, reward, done, info = np.zeros([640, 480, 3], dtype=np.uint8()), 20, False, {}
        return observation, reward, done, info

    def reset(self):
        observation = np.zeros([640, 480, 3], dtype=np.uint8())
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass


turn_off_log_warnings()

# Instantiate the env
env = CustomEev()

# To check that the environment follows the gym interface, using:
check_env(env)
print("Check success")

env = DummyVecEnv([lambda: env])
# Define and train the agent
model = A2C('CnnPolicy', env).learn(total_timesteps=100)
