import os

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from rlbaselines.common_utils.utils import evaluate


class EvalCallback(object):
    """
    Callback for evaluating an agent.
    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent, usually 5-20
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=20, log_dir='./model/gym/'):
        super(EvalCallback, self).__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.n_calls = 0
        self.best_mean_reward = -np.inf
        self.log_dir = log_dir

    def __call__(self, locals_, globals_):
        """
        This method will be called by the model.
        :param locals_: (dict)
        :param globals_: (dict)
        :return: (bool)
        """
        # Get the self object of the model
        self_ = locals_['self']

        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent:
            # you need to do self.n_eval_episodes loop using self.eval_env
            # hint: you can use self_.predict(obs)
            for i in range(self.n_eval_episodes):
                reward = evaluate(self_, self.eval_env, num_episodes=self.n_eval_episodes)

                # Save the agent and update self.best_mean_reward
                if self.best_mean_reward < reward:
                    self.best_mean_reward = reward

            print("Evaluation: Best mean reward in the evaluated env: {:.2f}, saved it.".format(self.best_mean_reward))
            self_.save(self.log_dir + 'best_evaluated_model_with_reward_' + str(self.best_mean_reward))

        self.n_calls += 1
        return True


if __name__ == '__main__':
    # Env used for training
    env = gym.make("CartPole-v1")
    env = DummyVecEnv([lambda: env])

    # Env for evaluating the agent
    eval_env = gym.make("CartPole-v1")
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create log dir
    log_dir = "./model/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Create the callback object
    callback = EvalCallback(eval_env, n_eval_episodes=5, eval_freq=20, log_dir=log_dir)

    # Create the RL model
    model = PPO2('MlpPolicy', env, verbose=0)

    # Train the RL model
    model.learn(int(100000), callback=callback, log_interval=1000)
