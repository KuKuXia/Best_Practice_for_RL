import numpy as np
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        return (np.random.normal(0.0, 0.1, size=(self.action_size,))).tolist()


obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.left_shoulder_camera.rgb = True
obs_config.right_shoulder_camera.rgb = True

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

task = env.get_task(ReachTarget)
training_steps = 120
episode_length = 40
obs = task.reset()

# agent = Agent(action_mode.action_size)

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: task])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action, _states = model.predict(obs)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()