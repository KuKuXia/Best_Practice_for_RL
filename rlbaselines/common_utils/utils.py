import logging
import os
import warnings

import gym
import numpy as np
import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecVideoRecorder


def turn_off_log_warnings():
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)

    tf.get_logger().setLevel(logging.ERROR)


def make_env(env_id, rank, seed=0):
    """
    Utility function for multi-processed env.
    :param env_id: (str) the environment ID
    :param rank: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :return: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def evaluate_multi_processes(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    env = model.get_env()
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        actions, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(actions)
        # Stats
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)

        mean_rewards = [0.0 for _ in range(env.num_envs)]
        n_episodes = 0
        for i in range(env.num_envs):
            mean_rewards[i] = np.mean(episode_rewards[i])
            n_episodes += len(episode_rewards[i])

        # Compute mean reward
        mean_reward = round(np.mean(mean_rewards), 1)
        print("Mean reward:", mean_reward, "Num episodes:", n_episodes)
        return mean_reward


def evaluate_last_100_episode(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL agent
    :param num_steps: (int) number of time-steps to evaluate it
    :return: (float) mean reward for the last 100 episode
    """
    env = model.get_env()
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # States are only useful when using LSTM policies
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        # Statics
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episode
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print('Mean reward of the last 100 episodes is: ', mean_100ep_reward, "Num episodes are: ", len(episode_rewards))
    return mean_100ep_reward


def evaluate(model, env, num_episodes=100):
    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward


def get_callback_vars(model, **kwargs):
    """
    Helps store variables for the callback functions
    :param model: (BaseRLModel)
    :param **kwargs: initial values of the callback variables
    """
    # save the called attribute in the model
    if not hasattr(model, "_callback_vars"):
        model._callback_vars = dict(**kwargs)
    else:  # check all the kwargs are in the callback variables
        for (name, val) in kwargs.items():
            if name not in model._callback_vars:
                model._callback_vars[name] = val
    return model._callback_vars  # return dict reference (mutable)


def record_video(env_id, model, video_length=500, prefix='', video_folder='./logs/videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """

    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(env_id, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()
