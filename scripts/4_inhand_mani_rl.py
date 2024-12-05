from myosuite.utils import gym
import isaacgym
from modexenvs.inhand_manipulation import *
from modexenvs.reach_envs import *

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import os
import numpy as np
from typing import Callable
from collections import deque as dq


class SaveSuccesses(BaseCallback):
    """
    sb3 callback used to calculate and monitor success statistics. Used in training functions.
    """
    def __init__(self, check_freq: int, log_dir: str, env_name: str, verbose: int = 1):
        super(SaveSuccesses, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'ignore')
        self.check_for_success = []
        self.success_buffer = dq(maxlen=100)
        self.success_results = []
        self.env_name = env_name

    def _on_rollout_start(self) -> None:
        self.check_for_success = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self) -> None:
        if sum(self.check_for_success) > 0:
            self.success_buffer.append(1)
        else:
            self.success_buffer.append(0)

        if len(self.success_buffer) > 0:
            self.success_results.append(sum(self.success_buffer)/len(self.success_buffer))
        self.logger.record('rollout/success', sum(self.success_buffer)/len(self.success_buffer))

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.check_for_success.append(self.locals['infos'][0]['solved'])
        return True

    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, f'success_{self.env_name}'), np.array(self.success_results))
#         plt.plot(range(len(self.success_results)), self.success_results)
#         plt.show()
        pass

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def train(env_name=None, policy_name=None, timesteps=1e7, seed=0):
    np.random.seed(int(seed))
    env = gym.make(env_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10.)

    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))

    model = SAC('MlpPolicy', env, learning_rate=linear_schedule(.001), buffer_size=int(3e5),
            learning_starts=1000, batch_size=256, tau=.02, gamma=.98, train_freq=(1, "episode"),
            gradient_steps=-1,policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f'./logs/in-hand_manipulation/{env_name}/{policy_name}_train_{seed}')
    # model = SAC('MlpPolicy', env, learning_rate=.0003, buffer_size=int(3e5),
    #         learning_starts=100, batch_size=256, tau=.005, gamma=.99, train_freq=(1, "episode"),
    #         gradient_steps=-1,policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f'./logs/in-hand_manipulation/{env_name}/{policy_name}_train_{seed}')
    
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ObjHoldFixed/")
    
    succ_callback = SaveSuccesses(check_freq=1, env_name=env_name+'_'+seed, 
                             log_dir=f'./logs/in-hand_manipulation/{env_name}/')
    
    save_callback = CheckpointCallback(save_freq=1e8, save_path=f'./logs/in-hand_manipulation/{env_name}/models/')
    
    callbacks = CallbackList([succ_callback, save_callback])
    # model.set_logger(configure(f'./logs/in-hand_manipulation/{env_name}/{policy_name}_results_{seed}'))

    model.learn(total_timesteps=int(timesteps), callback=callbacks, log_interval=4)

    model.save(f"./logs/in-hand_manipulation/{env_name}/{policy_name}_model_{env_name}_{seed}")
    env.save(f'./logs/in-hand_manipulation/{env_name}/{policy_name}_env_{env_name}_{seed}')

    # model.learn(total_timesteps=1e7)

if __name__ == '__main__':
    # train('myoHandReachOneStep', 'sac', 1e6, '0')
    # train('myoHandReachSequence', 'sac', 1e7, '0')
    # train('myoHandObjHoldFixedCustom', 'sac', 1e5, '3')
    # train('myoHandObjHoldRandomCustom', 'sac', 1e7, '4')
    # train('myoHandPenTwirlFixedCustom', 'sac', 3e6, '2')
    # train('myoHandPenTwirlRandomCustom', 'sac', 1e6, '3')
    # train('myoHandReorientCustom', 'sac', 1e7, '0')
    # train('myoHandReorient8Custom', 'sac', 1e7, '8')
    # train('myoHandReorient100Custom', 'sac', 1e7, '8')

    # train('myoHandReorientTarget2Custom', 'sac', 1e7, '0')
    train('myoHandBaodingCustom', 'sac', 1e7, '0')