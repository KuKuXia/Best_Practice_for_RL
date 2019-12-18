from rlbaselines.common_utils.utils import turn_off_log_warnings

turn_off_log_warnings()

import tensorflow as tf

import gym
from gym import spaces
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan


class NanAndInfEnv(gym.Env):
    """Custom Environment that raised NaNs and Infs"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(NanAndInfEnv, self).__init__()
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    def step(self, _action):
        randf = np.random.rand()
        if randf > 0.99:
            obs = float('NaN')
        elif randf > 0.98:
            obs = float('inf')
        else:
            obs = randf
        return [obs], 0.0, False, {}

    def reset(self):
        return [0.0]

    def render(self, mode='human', close=False):
        pass


def devision_by_zero_test():
    print("tensorflow test:")

    a = tf.constant(1.0)
    b = tf.constant(0.0)
    c = a / b

    sess = tf.Session()
    val = sess.run(c)  # this will be quiet
    print(val)
    sess.close()

    print("\r\nnumpy test:")

    a = np.float64(1.0)
    b = np.float64(0.0)
    val = a / b  # this will warn
    print(val)

    print("\r\npure python test:")

    a = 1.0
    b = 0.0
    val = a / b  # this will raise an exception and halt.
    print(val)


def numpy_test():
    np.seterr(all='raise')  # define before your code.

    # print("numpy devide by zeor test:")
    # a = np.float64(1.0)
    # b = np.float64(0.0)
    # val = a / b  # this will now raise an exception instead of a warning.
    # print(val)

    # print("numpy overflow test:")
    #
    # a = np.float64(10)
    # b = np.float64(1000)
    # val = a ** b  # this will now raise an exception
    # print(val)

    print("numpy propagation test:")

    a = np.float64('NaN')
    b = np.float64(1.0)
    val = a + b  # this will neither warn nor raise anything
    print(val)


def tensorflow_test():
    # print("tensorflow test:")
    #
    # a = tf.constant(1.0)
    # b = tf.constant(0.0)
    # c = a / b
    #
    # check_nan = tf.add_check_numerics_ops()  # add after your graph definition.
    #
    # sess = tf.Session()
    # val, _ = sess.run([c, check_nan])  # this will now raise an exception
    # print(val)
    # sess.close()

    # print("tensorflow overflow test:")
    #
    # check_nan = []  # the list of check_numerics operations
    #
    # a = tf.constant(10)
    # b = tf.constant(1000)
    # c = a ** b
    #
    # check_nan.append(tf.check_numerics(c, ""))  # check the 'c' operations
    #
    # sess = tf.Session()
    # val, _ = sess.run([c] + check_nan)  # this will now raise an exception
    # print(val)
    # sess.close()

    print("tensorflow propagation test:")

    check_nan = []  # the list of check_numerics operations

    a = tf.constant('NaN')
    b = tf.constant(1.0)
    c = a + b

    check_nan.append(tf.check_numerics(c, ""))  # check the 'c' operations

    sess = tf.Session()
    val, _ = sess.run([c] + check_nan)  # this will now raise an exception
    print(val)
    sess.close()


# numpy_test()
# tensorflow_test()

if __name__ == '__main__':
    # Create environment
    env = DummyVecEnv([lambda: NanAndInfEnv()])
    env = VecCheckNan(env, raise_exception=True)

    # Instantiate the agent
    model = PPO2('MlpPolicy', env)

    # Train the agent
    model.learn(
        total_timesteps=int(2e5))  # this will crash explaining that the invalid value originated from the environment.
    print("Training Ended.")
