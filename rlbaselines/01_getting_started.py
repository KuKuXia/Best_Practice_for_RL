import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1).learn(total_timesteps=10000, log_interval=1000)

obs = env.reset()
for i in range(2000):
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
