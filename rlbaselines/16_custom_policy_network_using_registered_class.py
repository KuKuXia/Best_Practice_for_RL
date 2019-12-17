from stable_baselines import A2C
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, LstmPolicy

from rlbaselines.common_utils.utils import turn_off_log_warnings


# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[256, 128, 64, dict(pi=[128, 128, 128],
                                                                        vf=[128, 128, 128])],
                                           feature_extraction="mlp")


# Here the net_arch parameter takes an additional (mandatory) ‘lstm’ entry within the shared network section.
# The LSTM is shared between value network and policy network.
class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


turn_off_log_warnings()

# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)

# Because the policy is now registered, you can pass
# a string to the agent constructor instead of passing a class
model = A2C(policy='CustomPolicy', env='LunarLander-v2', verbose=1).learn(total_timesteps=1000)

# Save the agent
model.save("./model/gym/a2c-lunar-registered-class")

del model
# When loading a model with a custom policy
# you MUST pass explicitly the policy when loading the saved model
model = A2C.load("./model/gym/a2c-lunar-registered-class", policy=CustomPolicy)

print('Trianing Ended')
