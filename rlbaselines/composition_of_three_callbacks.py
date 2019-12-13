import os

import gym
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from rlbaselines.auto_save_best_model_rlbaselines import auto_save_callback
from rlbaselines.best_evaluated_model import EvalCallback
from rlbaselines.common_utils.utils import get_callback_vars
from rlbaselines.live_plotting_using_rlbaselines import plotting_callback
from rlbaselines.progressbar_rlbaselines import progressbar_callback


def compose_callback(*callback_funcs):  # takes a list of functions, and returns the composed function.
    def _callback(_locals, _globals):
        continue_training = True
        for cb_func in callback_funcs:
            if cb_func(_locals, _globals) is False:  # as a callback can return None for legacy reasons.
                continue_training = False
        return continue_training

    return _callback


if __name__ == '__main__':
    # Create log dir
    log_dir = "./model/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('CartPole-v1')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO2('MlpPolicy', env, verbose=0)
    get_callback_vars(model, log_dir=log_dir)

    # Create the callback object
    eval_callback = EvalCallback(env, n_eval_episodes=5, eval_freq=50, log_dir=log_dir)

    with progressbar_callback(100000) as progress_callback:
        model.learn(100000,
                    callback=compose_callback(progress_callback, plotting_callback, auto_save_callback, eval_callback))
