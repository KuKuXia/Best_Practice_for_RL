import gym
import numpy as np
from gym import spaces
from stable_baselines import ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv


class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left
    """
    metadata = {'render_modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10):
        super(GoLeftEnv, self).__init__()

        # Size of the 1D-Grad
        self.grid_size = grid_size

        # Initialize the agent at the right of the grid
        self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # This can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(1,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = self.grid_size - 1
        # Here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(self.agent_pos).astype(np.float32)

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError('Recevied invalid action= {} which is not part of the actrion space'.format(action))

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = self.agent_pos == 0

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array(self.agent_pos).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        # Agent is represented as a cross, rest as a dot
        print('.' * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


def SimpleAgent():
    env = GoLeftEnv(grid_size=10)
    _ = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    go_left = 0
    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(go_left)
        print("obs= ", obs, "reward= ", reward, 'done= ', done, 'info: ', info)
        env.render()
        if done:
            print("Goal reached", 'reward= ', reward)
            break


def StableBaselinesAgent():
    env = GoLeftEnv(grid_size=10)
    env = Monitor(env, filename=None, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    # model = DQN('MlpPolicy', env, verbose=1).learn(5000, log_interval=500)
    # model = PPO2('MlpPolicy', env, verbose=1).learn(5000, log_interval=500)
    # model = A2C('MlpPolicy', env, verbose=1).learn(10000, log_interval=500)
    model = ACKTR('MlpPolicy', env, verbose=1).learn(10000, log_interval=500)

    # Test the trained agent
    n_steps = 20
    obs = env.reset()
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print('Step {}'.format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs= ', obs, 'reward= ', reward, 'done= ', done, 'info= ', info)
        env.render(mode='console')
        if done:
            print("Goal reached!", 'reward= ', reward)
            break


if __name__ == '__main__':
    # SimpleAgent()
    StableBaselinesAgent()
