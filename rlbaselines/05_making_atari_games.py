from stable_baselines import ACER
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack

from rlbaselines.common_utils.utils import turn_off_log_warnings

if __name__ == '__main__':
    # Turn off warnings
    turn_off_log_warnings()

    # There already exists an environment generator that will make and wrap atari environments correctly.
    env = make_atari_env('PongNoFrameskip-v4', num_env=4, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)

    model = ACER(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the model
    model.save('./model/gym/acer_pong.pkl')
    # Delete the model for demonstrate the load function
    del model

    trained_model = ACER.load("./model/gym/acer_pong.pkl", verbose=1)
    env = make_atari_env('PongNoFrameskip-v4', num_env=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    trained_model.set_env(env)

    # Continue training
    trained_model.learn(int(1e5))

    # Save the model again
    trained_model.save('./model/gym/acer_pong_continued.pkl')
    obs = env.reset()
    for i in range(2000):
        action, _states = trained_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()
