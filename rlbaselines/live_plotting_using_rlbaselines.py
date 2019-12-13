import gym
import matplotlib.pyplot as plt
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

from rlbaselines.common_utils.utils import *


def plotting_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # get callback variables, with default values if unintialized
    callback_vars = get_callback_vars(_locals["self"], plot=None)
    log_dir = callback_vars['log_dir']

    # get the monitor's data
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if callback_vars["plot"] is None:  # make the plot
        plt.ion()
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        line, = ax.plot(x, y)
        callback_vars["plot"] = (line, ax, fig)
        plt.show()
    else:  # update and rescale the plot
        callback_vars["plot"][0].set_data(x, y)
        callback_vars["plot"][-2].relim()
        callback_vars["plot"][-2].set_xlim([_locals["total_timesteps"] * -0.02,
                                            _locals["total_timesteps"] * 1.02])
        callback_vars["plot"][-2].autoscale_view(True, True, True)
        callback_vars["plot"][-1].canvas.draw()


if __name__ == '__main__':
    # Create log dir
    log_dir = "./model/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('MountainCarContinuous-v0')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO2('MlpPolicy', env, verbose=0)
    model.learn(20000, callback=plotting_callback)
