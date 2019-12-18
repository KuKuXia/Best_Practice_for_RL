from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()

from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='./model/expert_trajectories/expert_dqn_cartpole.npz',
                        traj_limitation=1, batch_size=128)

model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)

# Pre-train the PPO2 model
model.pretrain(dataset, n_epochs=10000)

# As an option, you can train the RL agent
model.learn(int(1e5))

# Test the pre-trained model
env = model.get_env()
obs = env.reset()

reward_sum = 0.0
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    env.render()
    if done:
        print("The reward is: ", reward_sum)
        reward_sum = 0.0
        obs = env.reset()
env.close()
print("Train Ended")
