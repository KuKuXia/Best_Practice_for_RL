from stable_baselines import TD3
from tqdm.auto import tqdm


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class progressbar_callback(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        def callback_progressbar(local_, global_):
            self.pbar.n = local_["self"].num_timesteps
            self.pbar.update(0)

        return callback_progressbar

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


if __name__ == '__main__':
    model = TD3('MlpPolicy', 'Pendulum-v0', verbose=0)
    with progressbar_callback(2000) as callback:  # this the garanties that the tqdm progress bar closes correctly
        model.learn(2000, callback=callback)
