import gym
import numpy as np
import tensorflow as tf
from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv

from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()

# 添加tensorboard_log参数即可集成tensorboard支持
# A2C的构造器可以直接根据字符串将环境进行矢量化，因为注册的原因
model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1, tensorboard_log='./model/tensorboard/sac_pendulum/')

# Define a new property to avoid global variable
model.is_tb_set = False


def callback(locals_, globals_):
    self_ = locals_['self']
    # Log additional tensor
    if not self_.is_tb_set:
        with self_.graph.as_default():
            tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
            self_.summary = tf.summary.merge_all()
        self_.is_tb_set = True
    # Log scalar value (here a random variable)
    value = np.random.random()
    summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
    locals_['writer'].add_summary(summary, self_.num_timesteps)
    return True


# 可以设置自定义的训练日志，默认为算法名称
model.learn(total_timesteps=100, tb_log_name='first_run', callback=callback)
# 默认情况下保存model不会保存tensorboard的路径地址
model.save('./model/gym/sac_pendulum.pkl')

# 为了展示load，删除model
del model

# The A2C algorithm require a vectorized environment ot run
env = gym.make('Pendulum-v0')
env = DummyVecEnv([lambda: env])

# 显式设置log路径
model = SAC.load('./model/gym/sac_pendulum.pkl', env=env, tensorboard_log='./model/tensorboard/sac_pendulum/')
print('Loaded model.')

# Define a new property to avoid global variable
model.is_tb_set = False

# Pass reset_num_timesteps=False to continue the training curve in tensorboard
# By default, it will create a new curve
model.learn(total_timesteps=10000, tb_log_name="second_run", callback=callback, reset_num_timesteps=False)
model.learn(total_timesteps=10000, tb_log_name="third_run", callback=callback, reset_num_timesteps=False)
model.save('./model/gym/sac_carpole_final.pkl')
print('Training Ended.')
env.close()
