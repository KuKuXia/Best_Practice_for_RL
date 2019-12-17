import argparse

import gym
from gym import wrappers, logger


class RandomAgent(object):
    """
    The world's simplest agent!
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    env = gym.make(args.env_id)

    outdir = '../model/gym/random-agent-results/'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False

    for i in range(episode_count):
        obs = env.reset()
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            if done:
                break

    env.close()
