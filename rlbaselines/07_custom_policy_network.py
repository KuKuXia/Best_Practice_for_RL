from stable_baselines import A2C
from stable_baselines.common.policies import FeedForwardPolicy

from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()


# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])],
                                           feature_extraction="mlp")


model = A2C(CustomPolicy, 'LunarLander-v2', verbose=1)
# Train the agent
model.learn(total_timesteps=100000)

env = model.get_env()
obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
