import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from myosuite.utils import gym
import isaacgym
from modexenvs.inhand_manipulation import *
from scripts.train_inhand_mani import InhandTrainer

from algs.CEM import CEMInhand
import time
from omegaconf import OmegaConf
from tqdm import tqdm
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import numpy as np

# set random seed
np.random.seed(0)

# task = 'myoHandObjHoldFixedCustom'
task = 'myoHandObjHoldRandomCustom'
exp_num = 100
success = 0

env = gym.make(task)
env.reset()
###### Experiments #######

# # Model-free Reinforcement learning policy
# # model = SAC.load('logs/in-hand_manipulation/myoHandObjHoldFixedCustom/models/rl_model_900000_steps.zip')
# # model = SAC.load('logs/in-hand_manipulation/myoHandPenTwirlFixedCustom/models/rl_model_1000000_steps.zip', env=env)
# model = SAC.load('logs/in-hand_manipulation/myoHandObjHoldRandomCustom/sac_model_myoHandObjHoldRandomCustom_3.zip')
# vec = VecNormalize.load(f'logs/in-hand_manipulation/myoHandObjHoldRandomCustom/sac_env_myoHandObjHoldRandomCustom_3', DummyVecEnv([lambda: env]))
# for i in range(exp_num):
#     obs = env.reset()
#     done = False
#     step = 0
#     while not done and step < 75:
#         obs = env.get_obs()
#         obs = vec.normalize_obs(obs)
#         action, _states = model.predict(obs,deterministic=False)
#         obs, reward, _, _, info = env.step(action)
#         done = info['solved']
#         step += 1
#     err = np.linalg.norm(info['obs_dict']['obj_err'])
#     # err = np.linalg.norm(info[0]['obs_dict']['obj_err_pos'])
#     print("="*5, f' solved: {done}, err: {err}', "="*5)
#     if done:
#         success += 1

# Model-based planning policy

# instance and load the model

# cfg = OmegaConf.load('models/cfg/inhand_mani/objectholdfixed.yaml')
# model = InhandTrainer(cfg)
# model.load_model('logs/models/objectholdfixed_Aug_14_11:12:33_2024/ckpt-1.pt')
# model.net.eval()

cfg = OmegaConf.load('models/cfg/inhand_mani/objectholdrandom.yaml')
model = InhandTrainer(cfg)
model.load_model('logs/models/objectholdrand_Aug_13_20:59:06_2024/ckpt-1.pt')
model.net.eval()

# CEM
cem_policy = CEMInhand(
    env, 
    forward_model=model, 
    sample_num=800, 
    num_elites=20, 
    max_timesteps=5,  
    gamma=0.8,
    # verbose=True,  
)

for i in range(exp_num):
    actions,error, solved = cem_policy.step()
    # solved, r, info = cem_policy.eval(actions)
    # obj_state = cem_policy.get_env_obs()[15:]
    # err = np.linalg.norm(obj_state-cem_policy.target)
    print("="*5, f' solved: {solved}, err: {error}', "="*5)
    if solved:
        success += 1


print('-'*10)
print(f"Success rate: {success}/{exp_num}")
print('-'*10)

env.close()



