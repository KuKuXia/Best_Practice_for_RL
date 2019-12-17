import tensorflow as tf
from stable_baselines import PPO2

from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()

# Custom MLP policy of two layers of size 32 each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
# Create the agent
model = PPO2("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
# Retrieve the environment
env = model.get_env()
# Train the agent
model.learn(total_timesteps=1000)
# Save the agent
model.save("./model/gym/ppo2-cartpole")

del model
# the policy_kwargs are automatically loaded
model = PPO2.load("./model/gym/ppo2-cartpole")

env.close()
print('Trianing Ended')
