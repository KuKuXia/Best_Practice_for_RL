"""
A UR3 moves using delta end effector pose control.
This script contains RL_in_Practice of:
    - IK calculations.
    - Joint movement by setting joint target positions.
"""
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.robots.arms.ur5 import UR5

SCENE_FILE = join(dirname(abspath(__file__)), './scenes/scene_ur5_reach_target.ttt')
DELTA = 0.01
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = UR5()

starting_joint_positions = agent.get_joint_positions()
pos, quat = agent.get_tip().get_position(), agent.get_tip().get_quaternion()


def move(index, delta):
    pos[index] += delta
    new_joint_angles = agent.solve_ik(pos, quaternion=quat)
    agent.set_joint_target_positions(new_joint_angles)
    pr.step()


[move(2, -DELTA) for _ in range(20)]
[move(1, -DELTA) for _ in range(20)]
[move(2, DELTA) for _ in range(10)]
[move(1, DELTA) for _ in range(20)]

pr.stop()
pr.shutdown()
