import gym
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv

from rlbaselines.common_utils.utils import evaluate_last_100_episode, turn_off_log_warnings

# Turn off the warnings
turn_off_log_warnings()

# Build the gym evvironment and model
env = gym.make("LunarLander-v2")
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

# Random agent, before training
mean_reward_before_training = evaluate_last_100_episode(model, num_steps=10000)

# Train the agent and save it
model.learn(total_timesteps=int(10e4), log_interval=20)
# Save the agent
model.save('./model/gym/dqn_lunar')

# Delete the model to demonstrate loading
del model

# Load the trained agent
model = DQN.load("./model/gym/dqn_lunar")
model.set_env(env)
# Evaluate the trained agent
mean_reward_after_training = evaluate_last_100_episode(model, num_steps=10000)
env.close()
