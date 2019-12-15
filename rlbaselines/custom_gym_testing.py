import gym

env = gym.make('CartPole-v1')

# Box(4,) means that it is a Vector with 4 components
print("Observation space: ", env.observation_space)
print("Shape is: ", env.observation_space.shape)

# Discrete(2) means that there is two discrete actions
print("Action space: ", env.action_space)
print("Shape is: ", env.action_space.shape)

# The reset method is called at the beginning of an episode
obs = env.reset()

# Sample a random action
action = env.action_space.sample()
print("Sampled action: ", action)

# Execute the action using step
obs, reward, done, info = env.step(action)

# Note that obs is a numpy array
# Info is an empty dict for now but can contain any debugging info
# Reward is a scalar
print("The observation is {}, its shape is {}, reward is {}, done: {}, info: {}".format(obs, obs.shape, reward, done,
                                                                                        info))
