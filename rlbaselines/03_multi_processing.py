import time

import gym
from stable_baselines import ACKTR
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from rlbaselines.common_utils.utils import evaluate_multi_processes, turn_off_log_warnings, make_env

# Turn off the warnings
turn_off_log_warnings()
if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = ACKTR("MlpPolicy", env, verbose=0)

    # Evaluate the un-trained, random agent
    mean_reward_before_training = evaluate_multi_processes(model, num_steps=1000)

    n_timesteps = 25000

    # Multi-processed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print("Took {:.2f}s for multi-processed version - {:.2f} FPS".format(total_time_multi,
                                                                         n_timesteps / total_time_multi))

    # Single Process RL Training
    single_process_model = ACKTR("MlpPolicy", DummyVecEnv([lambda: gym.make(env_id)]), verbose=0)

    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single,
                                                                        n_timesteps / total_time_single))

    print("Multi-processed training is {:.2f}x faster!".format(total_time_single / total_time_multi))
