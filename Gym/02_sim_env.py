"""
This code is copied from the gym repo, Link:
https://raw.githubusercontent.com/openai/gym/master/examples/scripts/sim_env
"""

import argparse
import itertools
import time
from builtins import input

import numpy as np
from gym import spaces, envs

parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("--mode", choices=["noop", "random", "human"],
                    default="random")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--fps", type=float)
parser.add_argument("--once", action="store_true")
parser.add_argument("--ignore_done", action="store_true")
args = parser.parse_args()

env = envs.make(args.env)
ac_space = env.action_space

fps = args.fps or env.metadata.get('video.frames_per_second') or 100
if args.max_steps == 0: args.max_steps = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']

while True:
    env.reset()
    env.render(mode='human')
    print("Starting a new trajectory")
    for t in range(args.max_steps) if args.max_steps else itertools.count():
        done = False
        if args.mode == "noop":
            if isinstance(ac_space, spaces.Box):
                a = np.zeros(ac_space.shape)
            elif isinstance(ac_space, spaces.Discrete):
                a = 0
            else:
                raise NotImplementedError("noop not implemented for class {}".format(type(ac_space)))
            time.sleep(1.0 / fps)
        elif args.mode == "random":
            a = ac_space.sample()
            time.sleep(1.0 / fps)
        elif args.mode == "human":
            a = input("type action from {0,...,%i} and press enter: " % (ac_space.n - 1))
            try:
                a = int(a)
            except ValueError:
                print("WARNING: ignoring illegal action '{}'.".format(a))
                a = 0
            if a >= ac_space.n:
                print("WARNING: ignoring illegal action {}.".format(a))
                a = 0
        _, _, done, _ = env.step(a)

        env.render()
        if done and not args.ignore_done:
            break
    print("Done after {} steps".format(t + 1))
    if args.once:
        break
    else:
        input("Press enter to continue")
