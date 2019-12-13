import gym
from stable_baselines import A2C
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

from rlbaselines.common_utils.utils import *


def auto_save_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # get callback variables, with default values if uninitialized
    callback_vars = get_callback_vars(_locals["self"], n_steps=0, best_mean_reward=-np.inf)
    log_dir = callback_vars['log_dir']
    # skip every 20 steps
    if callback_vars["n_steps"] % 20 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])

            # New best model, you could save the agent here
            if mean_reward > callback_vars["best_mean_reward"]:
                callback_vars["best_mean_reward"] = mean_reward
                # Example for saving best model
                print("Trinaing: Saving the best model at {} timesteps, current best mean reward is {}".format(x[-1],
                                                                                                               mean_reward))
                _locals['self'].save(log_dir + 'best_training_model_with_reward_{}'.format(mean_reward))
    callback_vars["n_steps"] += 1
    return True


if __name__ == '__main__':
    # Create log dir
    log_dir = "./model/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('CartPole-v1')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = A2C('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=10000, callback=auto_save_callback)
