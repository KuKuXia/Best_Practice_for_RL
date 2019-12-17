import gym
from gym import envs

# Algorithmic
# env = gym.make('Copy-v0')

# Atari
# env = gym.make('SpaceInvaders-v0')

# Box2d
# env = gym.make('LunarLander-v2')

# Classic Control
env = gym.make('CartPole-v0')


# MuJoCo
# env = gym.make('Humanoid-v2')

# Robotics
# env = gym.make('HandManipulateBlock-v0')

# Toy Text
# env = gym.make('FrozenLake-v0')

def list_envs():
    envids = [spec.id for spec in envs.registry.all()]
    for envid in sorted(envids):
        print(envid)


list_envs()

env.reset()
while True:
    env.step(env.action_space.sample())
    env.render()
env.close()
